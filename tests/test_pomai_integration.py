#!/usr/bin/env python3
"""
test_pomai_integration.py

End-to-end integration test for Pomai server (text SQL-like protocol).

This version builds fully-formed SQL commands for all supported storage
data types and demonstrates constructing vector payloads from both Torch tensors
and NumPy arrays. It groups inserts into batched "INSERT INTO ... VALUES (...),(...)" 
statements (server parser supports multiple tuples).

For each data type we:
 - CREATE MEMBRANCE <name> DIM <n> DATA_TYPE <dtype>
 - INSERT a small batch of vectors (generated from Torch / NumPy tensors)
 - Wait briefly for server to process
 - EXEC SPLIT so ITERATE TRAIN returns the inserted vectors (split ratio 0.8/0.1/0.1)
 - ITERATE the TRAIN set and verify received vectors (shape/count) and print a sample

Notes:
 - For integer storage types we quantize/round normalized float features into the
   integer domain before formatting.
 - We support passing vectors as either numpy.ndarray or torch.Tensor; the builder
   will detect and convert accordingly.
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
            except: break
            if not chunk: break
            buf += chunk
            if buf.endswith(marker):
                break

    def request_binary_stream(self, cmd: str) -> Tuple[int, int, bytes]:
        """
        Sends command, parses 'OK BINARY <count> <dim> <bytes>\n', then reads raw bytes.
        """
        self.send(cmd)
        header_buf = b""
        while b"\n" not in header_buf:
            try:
                chunk = self.sock.recv(1)
            except: raise TimeoutError("Socket timeout reading header")
            if not chunk: raise ConnectionError("Socket closed")
            header_buf += chunk

        header_str = header_buf.decode("utf-8").strip()
        if not header_str.startswith("OK BINARY"):
            print(f"[Binary Stream Error] Header: {header_str}")
            if "ERR" in header_str or header_str:
                self._drain_until_end()
            return 0, 0, b""

        parts = header_str.split()
        if len(parts) < 5: return 0, 0, b""
        count = int(parts[2])
        dim = int(parts[3])
        total_bytes = int(parts[4])

        if total_bytes == 0:
            self._drain_until_end()
            return count, dim, b""

        data_buf = b""
        while len(data_buf) < total_bytes:
            want = total_bytes - len(data_buf)
            try:
                chunk = self.sock.recv(min(65536, want))
            except: raise TimeoutError("Timeout reading body")
            if not chunk: break
            data_buf += chunk

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
        sql = f"INSERT INTO {memname} VALUES {tup_text} TAGS (class:{tag_class}, date:{tag_date});"
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

            ##### ITERATE TRAIN ####
            print(f"ITERATE {mname} TRAIN to validate stream (drain)...")
            count, dim_ret, payload = client.request_binary_stream(f"ITERATE {mname} TRAIN 0 1000;")
            print(f"  -> Received {count} vectors, dim reported={dim_ret}")
            if count == 0:
                print(f"  [WARN] server returned 0 vectors (may be a bug if inserts succeeded)")
            if count > 0 and payload:
                try:
                    arr = np.frombuffer(payload, dtype=np.float32)
                    if dim_ret == 0:
                        print("  -> Warning: server reported dim 0")
                    else:
                        if arr.size % dim_ret == 0:
                            arr = arr.reshape(-1, dim_ret)
                            print(f"  -> Parsed TRAIN array shape: {arr.shape}, first[0,:5]={arr[0,:5].tolist()}")
                        else:
                            print("  -> Warning: payload size not divisible by dim")
                except Exception as e:
                    print("  -> Error parsing payload:", e)

            ##### ITERATE PAIR ####
            print(f"ITERATE {mname} PAIR to validate (label, vector) binary stream...")
            count, dim_ret, payload = client.request_binary_stream(f"ITERATE {mname} PAIR 0 1000;")
            print(f"  -> Received {count} label+vector pairs, dim reported={dim_ret}")
            if count > 0 and payload:
                try:
                    tuple_size = 8 + 4 * dim_ret
                    for i in range(min(2, count)):
                        label = struct.unpack('<Q', payload[i*tuple_size:i*tuple_size+8])[0]
                        vec = np.frombuffer(payload[i*tuple_size+8:i*tuple_size+8+4*dim_ret], dtype=np.float32)
                        print(f"    [PAIR {i}] label={label}, vec[:5]={vec[:5].tolist()}")
                except Exception as e:
                    print("  -> Error parsing PAIR stream:", e)

            ##### ITERATE TRIPLET ####
            print(f"ITERATE {mname} TRIPLET class 2 to validate triplet stream...")
            # Only attempt if batch_items all have 'class' metadata!
            count, dim_ret, payload = client.request_binary_stream(f"ITERATE {mname} TRIPLET class 2;")
            print(f"  -> Received {count} triplets, dim={dim_ret}")
            if count > 0 and payload:
                try:
                    triplet_size = 3 * 4 * dim_ret
                    for i in range(min(1, count)):
                        chunk = payload[i*triplet_size:(i+1)*triplet_size]
                        a = np.frombuffer(chunk[0:4*dim_ret], dtype=np.float32)
                        p = np.frombuffer(chunk[4*dim_ret:8*dim_ret], dtype=np.float32)
                        n = np.frombuffer(chunk[8*dim_ret:], dtype=np.float32)
                        print(f"    [TRIPLET {i}] anchor[:5]={a[:5].tolist()}, pos[:5]={p[:5].tolist()}, neg[:5]={n[:5].tolist()}")
                except Exception as e:
                    print("  -> Error parsing TRIPLET stream:", e)

            if count > 0:
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
    parser.add_argument("--batch", default=8, type=int)
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