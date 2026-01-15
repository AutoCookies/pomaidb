#!/usr/bin/env python3
"""
Quick test harness for Pomai CLI.

Usage:
  python3 tests/test_pomai_cli.py --cli ./build/pomai_cli --host 127.0.0.1 --port 7777

This version improves prompt/readiness detection (accepts welcome banner).
"""

import argparse
import subprocess
import time
import sys
import socket
import threading

DEFAULT_CLI = "./build/pomai_cli"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7777
PROMPT_MARK = "[pomai"  # CLI prompt prefix
WELCOME_MARK = "PomaiDB CLI"  # banner printed on startup

def wait_server(host, port, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            time.sleep(0.1)
    return False

class CLIProcess:
    def __init__(self, path, host, port):
        self.path = path
        self.host = host
        self.port = port
        self.proc = None
        self._out_buf = []
        self._out_lock = threading.Lock()
        self._stop = False
        self._thread = None

    def start(self):
        cmd = [self.path, "-h", self.host, "-p", str(self.port)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        # start reader thread (read raw bytes so we capture prompt without newline)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # wait for readiness: either prompt or welcome banner
        return self.wait_for_ready(timeout=5.0)

    def _reader(self):
        try:
            # read raw bytes and append to buffer as text
            while True:
                b = self.proc.stdout.read(1)
                if not b:
                    break
                ch = b.decode(errors='ignore')
                with self._out_lock:
                    self._out_buf.append(ch)
        except Exception:
            pass

    def send(self, txt):
        if not self.proc or self.proc.stdin.closed:
            raise RuntimeError("CLI process not started or stdin closed")
        if not txt.endswith("\n"):
            txt = txt + "\n"
        try:
            self.proc.stdin.write(txt.encode())
            self.proc.stdin.flush()
        except Exception as e:
            raise

    def collect_recent(self, last_chars=4096):
        with self._out_lock:
            s = "".join(self._out_buf[-last_chars:])
        return s

    def wait_for_ready(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            out = self.collect_recent()
            if PROMPT_MARK in out or WELCOME_MARK in out:
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        try:
            if self.proc:
                try:
                    self.send("EXIT;")
                except Exception:
                    pass
                time.sleep(0.1)
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._stop = True

# ---- rest of test unchanged, using new CLIProcess ----

def assert_no_err(text, label):
    if "ERR" in text.upper() or "ERROR" in text.upper():
        print(f"[FAIL] {label}: server returned error snippet:")
        print(text.strip()[:1000])
        return False
    print(f"[OK] {label}")
    return True

def run_test(cli_path, host, port):
    print(f"[TEST] Checking server at {host}:{port} ...")
    if not wait_server(host, port, timeout=5.0):
        print("[FAIL] Pomai server not reachable. Start server first.")
        return 2

    print("[TEST] Starting CLI...")
    cli = CLIProcess(cli_path, host, port)
    if not cli.start():
        print("[FAIL] CLI did not indicate readiness in time. Dump output:")
        print(cli.collect_recent())
        return 3

    MEMBR = "py_cli_test_membr"
    dim = 8
    v = [0.1] * dim
    vec_str = ",".join(f"{x:.6f}" for x in v)

    steps_ok = True

    def send_and_wait(cmd, sleep_after=0.05):
        print(f">>> {cmd.strip()}")
        cli.send(cmd)
        time.sleep(sleep_after)
        out = cli.collect_recent()
        print("--- RESPONSE (tail) ---")
        print(out[-1000:])
        print("--- END ---")
        return out

    # drop (ignore errors)
    out = send_and_wait(f"DROP MEMBRANCE {MEMBR};")
    # create
    out = send_and_wait(f"CREATE MEMBRANCE {MEMBR} DIM {dim} RAM 16;")
    if not assert_no_err(out, "CREATE"):
        steps_ok = False

    # use
    out = send_and_wait(f"USE {MEMBR};")
    # insert via short syntax (since USE active)
    out = send_and_wait(f"INSERT VALUES (k_1, [{vec_str}]);")
    if not assert_no_err(out, "INSERT"):
        steps_ok = False

    # get label
    out = send_and_wait(f"GET LABEL k_1;")
    if not assert_no_err(out, "GET"):
        steps_ok = False

    # search
    out = send_and_wait(f"SEARCH QUERY ([{vec_str}]) TOP 1;")
    if "ERR" in out.upper() or "ERROR" in out.upper():
        print("[FAIL] SEARCH returned an error")
        steps_ok = False
    else:
        print("[OK] SEARCH executed (inspect server output above)")

    # cleanup
    out = send_and_wait(f"DROP MEMBRANCE {MEMBR};")
    if not assert_no_err(out, "DROP"):
        steps_ok = False

    cli.stop()
    return 0 if steps_ok else 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cli", default=DEFAULT_CLI, help="Path to pomai_cli binary")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", default=DEFAULT_PORT, type=int)
    args = p.parse_args()

    rc = run_test(args.cli, args.host, args.port)
    if rc == 0:
        print("[ALL OK] CLI basic smoke test passed")
    else:
        print("[FAILED] CLI smoke test failed (code {})".format(rc))
    sys.exit(rc)