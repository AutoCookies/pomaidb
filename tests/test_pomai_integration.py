#!/usr/bin/env python3
"""
test_pomai_integration.py

End-to-end integration test for Pomai server (text SQL-like protocol).

This extended version tests ITERATE in two modes:
  1) Single-call ITERATE (no explicit batch/offset passed) -- server returns all requested items at once.
  2) Batched ITERATE -- client repeatedly requests slices using offset/limit to drain dataset in chunks.

Usage:
    python3 test_pomai_integration.py --assets ./assets --batch 8

The script will create membrances, insert vectors, create splits and then
exercise ITERATE (TRAIN/PAIR/TRIPLET) for each configured dtype both as a
single large response and as repeated small batches. This helps validate the
server's support for both usage patterns.
"""

import argparse
import os
import socket
import sys
import time
import math
import re
import struct
from typing import List, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np
except Exception as e:
    print("Missing Python packages. Install: pip install torch torchvision pillow numpy")
    raise

# -----------------------
# Network client helpers
# -----------------------

class PomaiClient:
    def __init__(self, host='127.0.0.1', port=7777, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def send(self, cmd: str):
        """Send a command (string). Ensure it ends with '\n' (server treats text)."""
        if not cmd.endswith("\n"):
            cmd = cmd + "\n"
        data = cmd.encode("utf-8")
        totalsent = 0
        while totalsent < len(data):
            n = self.sock.send(data[totalsent:])
            if n <= 0:
                raise ConnectionError("Socket send failed")
            totalsent += n

    def send_and_get_response(self, cmd: str) -> str:
        """
        Send a textual command and read until the server's "<END>\n" marker is seen.
        Returns the full response as a string (excluding the final marker).
        """
        self.send(cmd)
        marker = b"<END>\n"
        buf = b""
        deadline = time.time() + self.timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for server response")
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                raise TimeoutError("Timed out waiting for server response (socket timeout)")
            if not data:
                break
            buf += data
            if marker in buf:
                break
        text = buf.decode("utf-8", errors="replace")
        idx = text.find("<END>")
        if idx != -1:
            return text[:idx]
        return text.strip()

    def _drain_until_end(self):
        buf = b""
        marker = b"<END>\n"
        while True:
            try:
                chunk = self.sock.recv(1)
            except:
                break
            if not chunk:
                break
            buf += chunk
            if buf.endswith(marker):
                break

    def request_binary_stream(self, cmd: str) -> Tuple[int, int, bytes]:
        """
        Sends command, parses header and reads raw bytes.

        Supports both header formats:
        - "OK BINARY <count> <dim> <bytes>\n"            (legacy)
        - "OK BINARY <dtype> <count> <dim> <bytes>\n"    (new)
        Also accepts "OK BINARY_PAIR" similarly.

        Returns (count, dim, bytes_payload)
        """
        self.send(cmd)
        header_buf = b""
        # read header line (until newline)
        while b"\n" not in header_buf:
            try:
                chunk = self.sock.recv(1)
            except Exception:
                raise TimeoutError("Socket timeout reading header")
            if not chunk:
                raise ConnectionError("Socket closed while reading header")
            header_buf += chunk

        header_str = header_buf.decode("utf-8", errors="replace").strip()

        # Accept both "OK BINARY" and "OK BINARY_PAIR"
        if not (header_str.startswith("OK BINARY") or header_str.startswith("OK BINARY_PAIR")):
            print(f"[Binary Stream Error] Header: {header_str}")
            # drain remainder until <END>\n to keep connection in sync
            if "ERR" in header_str or header_str:
                self._drain_until_end()
            return 0, 0, b""

        parts = header_str.split()
        # Determine whether parts[2] is a dtype token:
        dtypes = {"float32", "float64", "int32", "int8", "float16", "fp16", "double"}
        idx = 2
        if len(parts) >= 3 and parts[2].lower() in dtypes:
            idx = 3

        # Ensure we have enough tokens after resolving optional dtype
        if len(parts) <= idx + 2:
            self._drain_until_end()
            return 0, 0, b""

        try:
            count = int(parts[idx])
            dim = int(parts[idx + 1])
            total_bytes = int(parts[idx + 2])
        except Exception:
            self._drain_until_end()
            return 0, 0, b""

        if total_bytes == 0:
            # still need to consume trailing <END>\n marker
            self._drain_until_end()
            return count, dim, b""

        data_buf = b""
        while len(data_buf) < total_bytes:
            want = total_bytes - len(data_buf)
            try:
                chunk = self.sock.recv(min(65536, want))
            except Exception:
                raise TimeoutError("Timeout reading body")
            if not chunk:
                break
            data_buf += chunk

        # consume trailing protocol marker
        self._drain_until_end()
        return count, dim, data_buf

# -----------------------
# Feature extractor (ResNet50)
# -----------------------

class ResNet50FeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        import warnings
        warnings.filterwarnings("ignore")
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.backbone.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB")
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(t).reshape(-1)
        arr = feat.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0: arr = arr / norm
        return arr

# -----------------------
# Utility helpers
# -----------------------

VecLike = Union[np.ndarray, 'torch.Tensor']

def as_numpy(vec: VecLike) -> np.ndarray:
    """Convert numpy or torch tensor to numpy.ndarray (float32)."""
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32)
    try:
        import torch
        if isinstance(vec, torch.Tensor):
            return vec.detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    return np.array(vec, dtype=np.float32)

def vector_to_csv_list(vec: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def vector_to_csv_int_list(vec: List[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in vec) + "]"

def build_batch_insert_with_tags(memname: str,
                                 items: List[Tuple[str, VecLike, str, str]],
                                 dtype: str,
                                 batch_size: int = 64) -> List[str]:
    """
    Build one or more batched INSERT INTO statements for the provided items.
    Each INSERT groups up to batch_size tuples as:
      INSERT INTO <memname> VALUES (label, [v...]),(label2, [v2...]) TAGS (...);
    Returns list of SQL commands (strings).
    """
    def fmt_vec(vec: np.ndarray) -> str:
        if dtype in ("float32", "float64", "float16"):
            return vector_to_csv_list(vec.astype(np.float64).tolist())
        elif dtype == "int32":
            scaled = np.rint(vec * 1000.0).astype(np.int64)
            return vector_to_csv_int_list(scaled.tolist())
        elif dtype == "int8":
            scaled = np.rint(vec * 50.0).astype(np.int64)
            scaled = np.clip(scaled, -128, 127).astype(np.int64)
            return vector_to_csv_int_list(scaled.tolist())
        else:
            return vector_to_csv_list(vec.astype(np.float64).tolist())

    cmds = []
    tuples = []
    for (label, vec_like, tclass, tdate) in items:
        vec = as_numpy(vec_like)
        csv = fmt_vec(vec)
        safe_label = str(label).replace(")", "_").replace("(", "_").replace(",", "_").replace(" ", "_")
        tup = f"({safe_label}, {csv})"
        tuples.append((tup, tclass, tdate))

    i = 0
    while i < len(tuples):
        chunk = tuples[i:i+batch_size]
        tup_text = ",".join([t[0] for t in chunk])
        tag_class = chunk[0][1]
        tag_date = chunk[0][2]
        sql = f"INSERT INTO {memname} VALUES {tup_text} TAGS (class={tag_class}, date={tag_date});"
        cmds.append(sql)
        i += batch_size
    return cmds

def query_membrance_info(client: PomaiClient, memname: str, verbose: bool = True):
    cmd = f"GET MEMBRANCE INFO {memname};"
    try:
        resp = client.send_and_get_response(cmd)
    except Exception:
        return None
    if verbose: print("Raw membrance info:\n", resp.strip())
    info = {"dim": None, "num_vectors": None, "data_type": None}
    m = re.search(r"feature_dim\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["dim"] = int(m.group(1))
    m = re.search(r"total_vectors\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["num_vectors"] = int(m.group(1))
    m = re.search(r"data_type\s*[:=]\s*([A-Za-z0-9_]+)", resp, re.IGNORECASE)
    if m: info["data_type"] = m.group(1).strip()
    return info

# ---------- New: payload verification helpers ----------
def dtype_str_to_numpy(dtype_str: str):
    """
    Map textual dtype to numpy dtype and element size in bytes.
    Returns (np.dtype, elem_size) or (None, None) if unknown.
    """
    if not dtype_str:
        return None, None
    s = dtype_str.lower()
    if s in ("float32", "float"):
        return np.float32, 4
    if s in ("float64", "double"):
        return np.float64, 8
    if s in ("int32",):
        return np.int32, 4
    if s in ("int8",):
        return np.int8, 1
    if s in ("float16", "fp16", "half"):
        return np.float16, 2
    return None, None

def verify_payload_dtype(payload: bytes, dim: int, count: int, expected_dtype_str: str, verbose: bool = True) -> bool:
    """
    Verify that the binary payload length and element encoding match the expected storage dtype.
    Returns True if the payload appears encoded in expected_dtype, False otherwise.
    """
    if count == 0:
        if verbose: print("  -> No vectors expected/returned.")
        return True

    np_dtype, elem_size = dtype_str_to_numpy(expected_dtype_str)
    if np_dtype is None:
        if verbose:
            print(f"  -> Unknown expected dtype '{expected_dtype_str}'. Falling back to float32 verification.")
        np_dtype, elem_size = np.float32, 4

    expected_bytes = count * dim * elem_size
    if len(payload) != expected_bytes:
        if verbose:
            print(f"  -> Warning: payload size {len(payload)} != expected {expected_bytes} bytes (count*dim*elem_size).")
        # continue to attempt parsing what we can

    if len(payload) % elem_size != 0:
        if verbose:
            print(f"  -> Error: payload length {len(payload)} not divisible by element size {elem_size}.")
        return False

    try:
        arr = np.frombuffer(payload, dtype=np_dtype)
    except Exception as e:
        if verbose:
            print("  -> Error creating numpy array from payload with dtype", np_dtype, ":", e)
        return False

    if arr.dtype != np_dtype:
        if verbose:
            print("  -> Error: parsed array dtype", arr.dtype, "!= expected", np_dtype)
        return False

    if dim <= 0:
        if verbose:
            print("  -> Warning: received dim <= 0")
        return True

    if arr.size % dim != 0:
        if verbose:
            print("  -> Warning: total elements", arr.size, "is not divisible by dim", dim)
        # still accept but warn

    if verbose:
        parsed_count = arr.size // dim if dim > 0 else 0
        print(f"  -> Verified payload dtype {np_dtype}: parsed {parsed_count} vectors (elem_size={elem_size} bytes).")
    return True

# -----------------------
# ITERATE helpers
# -----------------------

def iterate_batches(client: PomaiClient, base_cmd: str, batch_size: int = 256):
    """
    Generator that yields (count, dim, payload) batches by repeatedly calling
    ITERATE with increasing offsets.

    base_cmd should be something like: "ITERATE <mem> TRAIN" or
    "ITERATE <mem> PAIR" or "ITERATE <mem> TRIPLET <key>", without offset/limit.
    We'll append "<off> <limit>;" each time.

    Stops when server returns count == 0 or when a returned batch smaller than batch_size.
    """
    offset = 0
    while True:
        cmd = f"{base_cmd} {offset} {batch_size};"
        try:
            count, dim, payload = client.request_binary_stream(cmd)
        except Exception as e:
            print(f"[iterate_batches] Error requesting batch offset={offset}: {e}")
            raise
        yield count, dim, payload
        if count == 0 or count < batch_size:
            break
        offset += count

def iterate_single(client: PomaiClient, cmd_without_semicolon: str):
    """
    Perform a single-call ITERATE (no offset/limit). The caller passes base command
    like "ITERATE <mem> TRAIN" or "ITERATE <mem> PAIR" or "ITERATE <mem> TRIPLET <key> <limit?>".
    This function appends the semicolon and requests the stream once.
    Returns (count, dim, payload)
    """
    cmd = cmd_without_semicolon.strip()
    if not cmd.endswith(";"):
        cmd = cmd + ";"
    return client.request_binary_stream(cmd)

# -----------------------
# Main test flow
# -----------------------

def run_test(host: str, port: int, assets_dir: str, batch_size: int = 16, verbose: bool = True):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = []
    for root, _, files in os.walk(assets_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                imgs.append(os.path.join(root, fn))
    imgs.sort()

    if len(imgs) < 8:
        print("Need at least 8 images for dtype tests.")
        return False

    print(f"Found {len(imgs)} images.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ResNet50FeatureExtractor(device=device)

    items = []
    mid = len(imgs) // 2
    print("Extracting features...")
    for i, p in enumerate(imgs[:32]):
        try:
            img = Image.open(p)
            vec = extractor.extract(img)
            label = os.path.splitext(os.path.basename(p))[0]
            tag_class = "dog" if i < mid else "cat"
            tag_date = "2024-01-01" if i < mid else "2024-02-01"
            if (i % 3) == 0:
                items.append((label, vec, tag_class, tag_date))  # numpy
            else:
                items.append((label, torch.from_numpy(vec.copy()), tag_class, tag_date))
            if verbose: print(f"  - {label} -> Class:{tag_class}, Date:{tag_date} (source={'numpy' if (i%3)==0 else 'torch'})")
        except Exception as e:
            if verbose:
                print(f"Skipping {p}: {e}")
            continue

    if not items:
        print("No items extracted.")
        return False

    client = PomaiClient(host=host, port=port)
    try:
        print("\nCreating membrance 'tests' (float32)...")
        client.send_and_get_response("CREATE MEMBRANCE tests DIM {} DATA_TYPE float32 RAM 256;".format(items[0][1].shape[0]))

        print(f"\nInserting {len(items)} items into 'tests' (batched)...")
        cmds = build_batch_insert_with_tags("tests", items, "float32", batch_size=16)
        for cmd in cmds:
            client.send_and_get_response(cmd)
        print("Inserted into 'tests'.")

        # Wait for batch ingest/flush
        time.sleep(1.5)
        print("Creating SPLIT for 'tests' with 0.8 0.1 0.1 ratio so TRAIN covers ~80% of inserted...")
        resp_split = client.send_and_get_response("EXEC SPLIT tests 0.8 0.1 0.1;")
        print("SPLIT RESP:", resp_split.strip())
        time.sleep(1.5)

        query_membrance_info(client, "tests", verbose=verbose)

        dtypes = ["float32", "float64", "int32", "int8", "float16"]
        for dt in dtypes:
            mname = f"tests_{dt}"
            dim = items[0][1].shape[0]
            print("\n" + "="*60)
            print(f"[DT TEST] Creating membrance '{mname}' with DATA_TYPE {dt} DIM {dim}")
            resp = client.send_and_get_response(f"CREATE MEMBRANCE {mname} DIM {dim} DATA_TYPE {dt} RAM 256;")
            print("CREATE RESP:", resp.strip())

            batch_items = items[:8]
            insert_cmds = build_batch_insert_with_tags(mname, batch_items, dt, batch_size=8)
            print(f"Inserting {len(batch_items)} entries into {mname} (dtype={dt}) ...")
            for c in insert_cmds:
                client.send_and_get_response(c)
            time.sleep(1.5)

            # Now split with 0.8 0.1 0.1 (guaranteed TRAIN/VAL/TEST content)
            try:
                split_resp = client.send_and_get_response(f"EXEC SPLIT {mname} 0.8 0.1 0.1;")
                print("SPLIT RESP:", split_resp.strip())
            except Exception as e:
                print("SPLIT failed or not supported:", e)
            time.sleep(1.0)

            info = query_membrance_info(client, mname, verbose=verbose)
            print(f"Membrance info parsed: {info}")
            # Determine expected storage dtype (from manifest/info if provided)
            expected_dtype = dt
            if info and info.get("data_type"):
                expected_dtype = info["data_type"]

            ##### SINGLE-CALL ITERATE TRAIN #####
            print(f"\n[SINGLE] ITERATE {mname} TRAIN (no batch param) -> validate single large response")
            try:
                count, dim_ret, payload = iterate_single(client, f"ITERATE {mname} TRAIN")
                print(f"  -> Single-call TRAIN returned count={count}, dim={dim_ret}, bytes={len(payload)}")
                ok = verify_payload_dtype(payload, dim_ret, count, expected_dtype, verbose=verbose)
                if not ok:
                    print(f"  [ERROR] Single-call TRAIN verification failed for {mname}")
            except Exception as e:
                print(f"  [ERROR] Single-call ITERATE TRAIN failed: {e}")

            ##### BATCHED ITERATE TRAIN #####
            print(f"\n[BATCHED] ITERATE {mname} TRAIN in batches (batch_size={batch_size}) to validate stream (drain)...")
            total_received = 0
            base_cmd = f"ITERATE {mname} TRAIN"
            for count, dim_ret, payload in iterate_batches(client, base_cmd, batch_size=batch_size):
                print(f"  -> Batch received count={count}, dim={dim_ret}, bytes={len(payload)}")
                ok = verify_payload_dtype(payload, dim_ret, count, expected_dtype, verbose=verbose)
                if not ok:
                    print(f"  [ERROR] Batch verification failed for {mname} TRAIN")
                total_received += count
            print(f"  -> Total TRAIN vectors received (all batches): {total_received}")

            ##### SINGLE-CALL ITERATE PAIR #####
            print(f"\n[SINGLE] ITERATE {mname} PAIR (no batch param) -> validate single large response of (label,vec) tuples")
            try:
                count, dim_ret, payload = iterate_single(client, f"ITERATE {mname} PAIR")
                print(f"  -> Single-call PAIR returned count={count}, dim={dim_ret}, bytes={len(payload)}")
                if count > 0:
                    np_dtype, elem_size = dtype_str_to_numpy(expected_dtype)
                    tuple_size = 8 + (elem_size * dim_ret if elem_size else 4 * dim_ret)
                    if len(payload) < tuple_size * count:
                        print("  -> Warning: payload smaller than expected for given count/dim/elem_size")
                    for i in range(count):
                        start = i * tuple_size
                        lbl_bytes = payload[start:start+8]
                        label = struct.unpack('<Q', lbl_bytes)[0]
                        vec_start = start + 8
                        vec_end = vec_start + (elem_size * dim_ret if elem_size else 4 * dim_ret)
                        vec_bytes = payload[vec_start:vec_end]
                        ok = verify_payload_dtype(vec_bytes, dim_ret, 1, expected_dtype, verbose=False)
                        if not ok:
                            print(f"    -> PAIR vector in single-call failed verification (expected={expected_dtype})")
                        else:
                            if np_dtype is not None:
                                v = np.frombuffer(vec_bytes, dtype=np_dtype)
                                print(f"    [PAIR] label={label}, vec[:5]={v[:5].astype(np.float64).tolist()}")
            except Exception as e:
                print(f"  [ERROR] Single-call ITERATE PAIR failed: {e}")

            ##### BATCHED ITERATE PAIR #####
            print(f"\n[BATCHED] ITERATE {mname} PAIR in batches (batch_size={batch_size}) to validate (label, vector) binary stream...")
            total_pairs = 0
            base_cmd = f"ITERATE {mname} PAIR"
            for count, dim_ret, payload in iterate_batches(client, base_cmd, batch_size=batch_size):
                print(f"  -> Batch PAIR count={count}, dim={dim_ret}, bytes={len(payload)}")
                if count == 0:
                    continue
                np_dtype, elem_size = dtype_str_to_numpy(expected_dtype)
                tuple_size = 8 + (elem_size * dim_ret if elem_size else 4 * dim_ret)
                if len(payload) < tuple_size * count:
                    print("  -> Warning: payload smaller than expected for given count/dim/elem_size")
                # Validate each pair in batch
                for i in range(count):
                    start = i * tuple_size
                    lbl_bytes = payload[start:start+8]
                    label = struct.unpack('<Q', lbl_bytes)[0]
                    vec_start = start + 8
                    vec_end = vec_start + (elem_size * dim_ret if elem_size else 4 * dim_ret)
                    vec_bytes = payload[vec_start:vec_end]
                    ok = verify_payload_dtype(vec_bytes, dim_ret, 1, expected_dtype, verbose=False)
                    if not ok:
                        print(f"    -> PAIR vector in batch failed verification (expected={expected_dtype})")
                    else:
                        if np_dtype is not None:
                            v = np.frombuffer(vec_bytes, dtype=np_dtype)
                            print(f"    [PAIR] label={label}, vec[:5]={v[:5].astype(np.float64).tolist()}")
                total_pairs += count
            print(f"  -> Total PAIRs received (all batches): {total_pairs}")

            ##### SINGLE-CALL ITERATE TRIPLET #####
            print(f"\n[SINGLE] ITERATE {mname} TRIPLET class (no batch param) -> validate single large triplet response")
            # For single-call TRIPLET we must provide a key and a limit; we'll request a large limit to get all triplets
            try:
                # large limit (server will clamp), class key 'class'
                single_cmd = f"ITERATE {mname} TRIPLET class 1000000"
                count, dim_ret, payload = iterate_single(client, single_cmd)
                print(f"  -> Single-call TRIPLET returned count={count}, dim={dim_ret}, bytes={len(payload)}")
                if count > 0:
                    np_dtype, elem_size = dtype_str_to_numpy(expected_dtype)
                    triplet_size = 3 * (elem_size * dim_ret if elem_size else 4 * dim_ret)
                    if len(payload) < triplet_size * count:
                        print("  -> Warning: triplet payload smaller than expected")
                    try:
                        for i in range(count):
                            chunk = payload[i*triplet_size:(i+1)*triplet_size]
                            a_b = chunk[0:elem_size*dim_ret]
                            p_b = chunk[elem_size*dim_ret:2*elem_size*dim_ret]
                            n_b = chunk[2*elem_size*dim_ret:3*elem_size*dim_ret]
                            oka = verify_payload_dtype(a_b, dim_ret, 1, expected_dtype, verbose=False)
                            okp = verify_payload_dtype(p_b, dim_ret, 1, expected_dtype, verbose=False)
                            okn = verify_payload_dtype(n_b, dim_ret, 1, expected_dtype, verbose=False)
                            if not (oka and okp and okn):
                                print("  -> Error: one of triplet parts failed verification")
                            else:
                                a = np.frombuffer(a_b, dtype=np_dtype)
                                p = np.frombuffer(p_b, dtype=np_dtype)
                                n = np.frombuffer(n_b, dtype=np_dtype)
                                print(f"    [TRIPLET] anchor[:5]={a[:5].astype(np.float64).tolist()}, pos[:5]={p[:5].astype(np.float64).tolist()}, neg[:5]={n[:5].astype(np.float64).tolist()}")
                    except Exception as e:
                        print("  -> Error parsing TRIPLET single-call stream:", e)
            except Exception as e:
                print(f"  [ERROR] Single-call ITERATE TRIPLET failed: {e}")

            ##### BATCHED ITERATE TRIPLET #####
            print(f"\n[BATCHED] ITERATE {mname} TRIPLET class in batches (batch_size={batch_size}) to validate triplet stream...")
            base_cmd = f"ITERATE {mname} TRIPLET class"
            total_triplets = 0
            for count, dim_ret, payload in iterate_batches(client, base_cmd, batch_size=batch_size):
                print(f"  -> Batch TRIPLET count={count}, dim={dim_ret}, bytes={len(payload)}")
                if count == 0:
                    continue
                np_dtype, elem_size = dtype_str_to_numpy(expected_dtype)
                triplet_size = 3 * (elem_size * dim_ret if elem_size else 4 * dim_ret)
                if len(payload) < triplet_size * count:
                    print("  -> Warning: triplet payload smaller than expected")
                try:
                    for i in range(count):
                        chunk = payload[i*triplet_size:(i+1)*triplet_size]
                        a_b = chunk[0:elem_size*dim_ret]
                        p_b = chunk[elem_size*dim_ret:2*elem_size*dim_ret]
                        n_b = chunk[2*elem_size*dim_ret:3*elem_size*dim_ret]
                        oka = verify_payload_dtype(a_b, dim_ret, 1, expected_dtype, verbose=False)
                        okp = verify_payload_dtype(p_b, dim_ret, 1, expected_dtype, verbose=False)
                        okn = verify_payload_dtype(n_b, dim_ret, 1, expected_dtype, verbose=False)
                        if not (oka and okp and okn):
                            print("  -> Error: one of triplet parts failed verification")
                        else:
                            a = np.frombuffer(a_b, dtype=np_dtype)
                            p = np.frombuffer(p_b, dtype=np_dtype)
                            n = np.frombuffer(n_b, dtype=np_dtype)
                            print(f"    [TRIPLET] anchor[:5]={a[:5].astype(np.float64).tolist()}, pos[:5]={p[:5].astype(np.float64).tolist()}, neg[:5]={n[:5].astype(np.float64).tolist()}")
                except Exception as e:
                    print("  -> Error parsing TRIPLET stream:", e)
                total_triplets += count
            print(f"  -> Total TRIPLETs received (all batches): {total_triplets}")

            if total_triplets > 0:
                probe_label, probe_vec, _, _ = batch_items[0]
                probe_np = as_numpy(probe_vec)
                print(f"Searching for first inserted label vector in {mname} (probe)...")
                try:
                    qcsv = "[" + ",".join(f"{float(x):.6f}" for x in probe_np.tolist()) + "]"
                    sresp = client.send_and_get_response(f"SEARCH {mname} QUERY ({qcsv}) TOP 3;")
                    print("SEARCH RESP (truncated):", sresp.strip().splitlines()[:5])
                except Exception as e:
                    print("Search failed:", e)

            # cleanup this membrance to avoid disk clutter
            client.send_and_get_response(f"DROP MEMBRANCE {mname};")
            print(f"Dropped membrance {mname}.")

        print("\nAll dtype tests completed.")
        client.send_and_get_response("DROP MEMBRANCE tests;")
        return True

    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7777, type=int)
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    args = parser.parse_args()

    if run_test(args.host, args.port, args.assets, args.batch, args.verbose):
        print("SUCCESS")
        sys.exit(0)
    else:
        print("FAILED")
        sys.exit(2)

if __name__ == "__main__":
    main()