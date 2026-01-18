#!/usr/bin/env python3
"""
test_pomai_integration.py

End-to-end integration test for Pomai server (text SQL-like protocol).

Features Tested:
- Batch Insert with TAGS (Class & Date)
- GET MEMBRANCE INFO (AI Contract)
- EXEC SPLIT (Random, Stratified, Cluster, Temporal)
- ITERATE Binary Stream (Train/Test & TRIPLET generation)
- ITERATE PAIR (label + vector)
- Search & Delete
"""

import argparse
import os
import socket
import sys
import time
import math
import re
import struct
from typing import List, Tuple

# Feature extraction deps
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np
except Exception as e:
    print("Missing Python packages. Install: pip install torch torchvision pillow numpy")
    raise

# --- Hashing Helper (Must match Server's FNV1a) ---
def fnv1a_hash(text: str) -> int:
    """Standard FNV-1a 64-bit hash used by PomaiDB server."""
    h = 14695981039346656037
    for char in text.encode("utf-8"):
        h ^= char
        h *= 1099511628211
        h &= 0xFFFFFFFFFFFFFFFF # Clamp to 64-bit unsigned
    return h

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
        
        # Read until marker
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
        # strip everything after marker
        idx = text.find("<END>")
        if idx != -1:
            return text[:idx]
        return text.strip()

    def _drain_until_end(self):
        """Robustly consume bytes until <END> marker."""
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

    # Binary Stream Handler
    def request_binary_stream(self, cmd: str) -> Tuple[int, int, bytes]:
        """
        Sends command, parses 'OK BINARY <count> <dim> <bytes>\n', then reads raw bytes.
        """
        self.send(cmd)
        
        # 1. Read Header
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

        # 2. Read Payload
        data_buf = b""
        while len(data_buf) < total_bytes:
            want = total_bytes - len(data_buf)
            try:
                chunk = self.sock.recv(min(65536, want))
            except: raise TimeoutError("Timeout reading body")
            if not chunk: break
            data_buf += chunk
            
        # 3. Consume Footer
        self._drain_until_end()

        return count, dim, data_buf

# -----------------------
# Feature extractor
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

def vector_to_csv_list(vec: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def make_insert_batch_cmd_with_tags(memname: str, items: List[Tuple[str, np.ndarray, str, str]]) -> List[str]:
    """
    Build INSERT command with TAGS (Class + Date).
    Item format: (label, vector, class_tag, date_tag)
    """
    cmds = []
    for label, vec, tag_class, tag_date in items:
        csv = vector_to_csv_list(vec.tolist())
        safe_label = label.replace(")", "_").replace("(", "_").replace(",", "_")
        # INSERT ... TAGS (class:..., date:...)
        cmd = f"INSERT INTO {memname} VALUES ({safe_label}, {csv}) TAGS (class:{tag_class}, date:{tag_date});"
        cmds.append(cmd)
    return cmds

def make_search_cmd(memname: str, query_vec: np.ndarray, topk: int = 5) -> str:
    qcsv = vector_to_csv_list(query_vec.tolist())
    return f"SEARCH {memname} QUERY ({qcsv}) TOP {topk};"

def query_membrance_info(client: PomaiClient, memname: str, verbose: bool = True):
    cmd = f"GET MEMBRANCE INFO {memname};"
    try:
        resp = client.send_and_get_response(cmd)
    except: return None
    
    if verbose: print("Raw membrance info:\n", resp.strip())
    
    info = {"dim": None, "num_vectors": None}
    m = re.search(r"dim\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["dim"] = int(m.group(1))
    m = re.search(r"num.*vectors\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["num_vectors"] = int(m.group(1))
    return info

# -----------------------
# Integration flow
# -----------------------

def run_test(host: str, port: int, assets_dir: str, batch_size: int = 16, verbose: bool = True):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = []
    for root, _, files in os.walk(assets_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                imgs.append(os.path.join(root, fn))
    imgs.sort()
    
    if len(imgs) < 4:
        print("Need at least 4 images.")
        return False

    print(f"Found {len(imgs)} images.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ResNet50FeatureExtractor(device=device)

    # Prepare Items with Mock Tags
    # First half -> Class:Dog, Date:2024-01-01
    # Second half -> Class:Cat, Date:2024-02-01
    items = []
    mid = len(imgs) // 2
    print("Extracting features...")
    for i, p in enumerate(imgs):
        try:
            img = Image.open(p)
            vec = extractor.extract(img)
            label = os.path.splitext(os.path.basename(p))[0]
            
            tag_class = "dog" if i < mid else "cat"
            tag_date = "2024-01-01" if i < mid else "2024-02-01"
            
            items.append((label, vec, tag_class, tag_date))
            if verbose: print(f"  - {label} -> Class:{tag_class}, Date:{tag_date}")
        except: pass

    if not items: return False

    client = PomaiClient(host=host, port=port)
    try:
        print("\nCreating membrance 'tests'...")
        client.send_and_get_response("CREATE MEMBRANCE tests DIM 2048 RAM 256;")

        print(f"\nInserting {len(items)} items with TAGS...")
        cmds = make_insert_batch_cmd_with_tags("tests", items)
        for cmd in cmds:
            client.send_and_get_response(cmd)

        print("Inserted.")
        
        # [FIXED] Wait for async flush
        print("Waiting 2s for server async flush...")
        time.sleep(2.0)
        
        query_membrance_info(client, "tests", verbose=verbose)

        # ---------------------------------------------------------
        # TEST 1: RANDOM SPLIT
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("[TEST 1] RANDOM Split (80/10/10)...")
        print(client.send_and_get_response("EXEC SPLIT tests 0.8 0.1 0.1;"))

        # ---------------------------------------------------------
        # TEST 2: STRATIFIED SPLIT (By Class)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 2] STRATIFIED Split by 'class' (50/25/25)...")
        resp = client.send_and_get_response("EXEC SPLIT tests 0.5 0.25 0.25 STRATIFIED class;")
        print(f"Response: {resp.strip()}")
        
        # ---------------------------------------------------------
        # TEST 3: CLUSTER SPLIT (Spatial)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 3] CLUSTER Split (50/0/50)...")
        resp = client.send_and_get_response("EXEC SPLIT tests 0.5 0.0 0.5 CLUSTER;")
        print(f"Response: {resp.strip()}")

        # ---------------------------------------------------------
        # TEST 4: TEMPORAL SPLIT (By Date)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 4] TEMPORAL Split by 'date' (50/0/50)...")
        resp = client.send_and_get_response("EXEC SPLIT tests 0.5 0.0 0.5 TEMPORAL date;")
        print(f"Response: {resp.strip()}")

        # ---------------------------------------------------------
        # TEST 5: STREAMING (Basic)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 5] Streaming TRAIN set via ITERATE...")
        count, dim, _ = client.request_binary_stream("ITERATE tests TRAIN 0 10000;")
        print(f"  -> Streamed {count} vectors from Train.")

        print("\n[TEST 5] Streaming TEST set via ITERATE...")
        count2, _, _ = client.request_binary_stream("ITERATE tests TEST 0 10000;")
        print(f"  -> Streamed {count2} vectors from Test.")

        # ---------------------------------------------------------
        # TEST 5b: ITERATE PAIR (label + vector)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 5b] ITERATE PAIR (label + vector) from TRAIN...")
        cnt_pair, dim_pair, raw_pair = client.request_binary_stream("ITERATE tests PAIR TRAIN 0 10000;")
        print(f"  -> Received {cnt_pair} pairs, dim={dim_pair}")
        if cnt_pair > 0 and raw_pair:
            mv = memoryview(raw_pair)
            parsed = []
            off = 0
            expect_per = 8 + dim_pair * 4
            for i in range(cnt_pair):
                if off + 8 > len(mv):
                    break
                label = int.from_bytes(mv[off:off+8], 'little')
                off += 8
                vec_bytes = mv[off: off + dim_pair*4]
                if len(vec_bytes) < dim_pair*4:
                    break
                vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                off += dim_pair*4
                parsed.append((label, vec))
            if parsed:
                lbl0, v0 = parsed[0]
                print(f"  -> First pair label(hash)={lbl0}, vec[:5]={v0[:5]}")
            else:
                print("  -> No valid pairs parsed (unexpected)")

        # ---------------------------------------------------------
        # [NEW] TEST 6: TRIPLET STREAMING (Generator)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 6] Streaming TRIPLET batch (Anchor, Positive, Negative)...")
        # Request 16 triplets based on 'class' tag
        triplet_cmd = "ITERATE tests TRIPLET class 16;" 
        count_tri, dim_tri, raw_tri = client.request_binary_stream(triplet_cmd)
        
        if count_tri > 0:
            # Each item is actually 3 vectors (A, P, N) flattened
            # Total floats = count_tri * 3 * dim_tri
            arr_tri = np.frombuffer(raw_tri, dtype=np.float32)
            expected_floats = count_tri * 3 * dim_tri
            
            if arr_tri.size == expected_floats:
                arr_tri = arr_tri.reshape(count_tri, 3, dim_tri)
                print(f"  -> Received TRIPLET batch: Shape={arr_tri.shape}")
                print(f"  -> Anchor[0][:5]:   {arr_tri[0,0,:5]}")
                print(f"  -> Positive[0][:5]: {arr_tri[0,1,:5]}")
                print(f"  -> Negative[0][:5]: {arr_tri[0,2,:5]}")
                
                # Basic check: A should not equal N (in most cases)
                dist_ap = np.linalg.norm(arr_tri[0,0] - arr_tri[0,1])
                dist_an = np.linalg.norm(arr_tri[0,0] - arr_tri[0,2])
                print(f"  -> Dist(A,P)={dist_ap:.4f}, Dist(A,N)={dist_an:.4f}")
            else:
                print(f"  -> ERROR: Shape mismatch. Got {arr_tri.size} floats, expected {expected_floats}")
        else:
            print("  -> WARNING: No triplets received (Check if 'class' tag exists and has >1 items per class).")

        print("="*50 + "\n")

        # Search Check
        probe_label, probe_vec, _, _ = items[0]
        safe_lbl = probe_label.replace(")", "_").replace("(", "_").replace(",", "_")
        print(f"Searching for {safe_lbl}...")
        print(client.send_and_get_response(make_search_cmd("tests", probe_vec)))

        # Cleanup
        print("\nCleanup...")
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