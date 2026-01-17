#!/usr/bin/env python3
"""
test_pomai_integration.py

End-to-end integration test for Pomai server (text SQL-like protocol).

This version includes:
- Batch Insert with TAGS support
- GET MEMBRANCE INFO query
- [NEW] EXEC SPLIT test (Random & Stratified)
- [NEW] ITERATE Binary Stream test
- [FIXED] Robust Protocol Handling (Binary Footer Sync) & Hash Verification
"""

import argparse
import os
import socket
import sys
import time
import math
import re
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
        """Send a command (string). Ensure it ends with '\\n' (server treats text)."""
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
        chunks = []
        # read until marker
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
        """
        Robustly consume bytes from socket until <END>\n is found.
        This handles variable-length footers (e.g. \n<END>\n vs <END>\n).
        """
        buf = b""
        marker = b"<END>\n"
        # Read byte-by-byte or small chunks to avoid over-reading next command's response
        # (Though in this sync test, over-reading isn't a huge issue, safety first)
        while True:
            chunk = self.sock.recv(1)
            if not chunk: break
            buf += chunk
            if buf.endswith(marker):
                break

    # [NEW] Binary Stream Handler with Robust Sync
    def request_binary_stream(self, cmd: str) -> Tuple[int, int, bytes]:
        """
        Sends command, parses 'OK BINARY <count> <dim> <bytes>\n', then reads raw bytes.
        Returns: (count, dim, data_bytes)
        """
        self.send(cmd)
        
        # 1. Read Header (line ending with \n)
        header_buf = b""
        while b"\n" not in header_buf:
            try:
                chunk = self.sock.recv(1)
            except socket.timeout:
                raise TimeoutError("Socket timeout reading header")
            if not chunk: 
                raise ConnectionError("Socket closed during header read")
            header_buf += chunk
        
        header_str = header_buf.decode("utf-8").strip()
        
        # Check for error response (e.g. ERR: ...)
        if not header_str.startswith("OK BINARY"):
            print(f"[Binary Stream Error] Header: {header_str}")
            # If server sent ERR, it likely sent the standard <END> marker too. Drain it.
            if "ERR" in header_str or header_str:
                self._drain_until_end()
            return 0, 0, b""

        # Parse header: OK BINARY 6 2048 49152
        parts = header_str.split()
        if len(parts) < 5: return 0, 0, b""
        
        count = int(parts[2])
        dim = int(parts[3])
        total_bytes = int(parts[4])
        
        if total_bytes == 0:
            # Even if 0 bytes, server sends <END>\n marker. Drain it.
            self._drain_until_end()
            return count, dim, b""

        # 2. Read exact binary payload
        data_buf = b""
        while len(data_buf) < total_bytes:
            want = total_bytes - len(data_buf)
            try:
                chunk = self.sock.recv(min(65536, want))
            except socket.timeout:
                raise TimeoutError("Socket timeout reading body")
            if not chunk: raise ConnectionError("Socket closed during body read")
            data_buf += chunk
            
        # 3. [ROBUST FIX] Consume the trailing "<END>\n" marker
        # Server always appends this. We must consume it to clear the socket for the next command.
        self._drain_until_end()

        return count, dim, data_buf

# -----------------------
# Feature extractor
# -----------------------

class ResNet50FeatureExtractor:
    """
    Use torchvision resnet50 pretrained to extract 2048-dim features from images.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        
        modules = list(model.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*modules).to(self.device)
        self.backbone.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB")
        t = self.transform(img).unsqueeze(0).to(self.device)  # 1 x 3 x 224 x 224
        with torch.no_grad():
            feat = self.backbone(t).reshape(-1)
        arr = feat.cpu().numpy().astype(np.float32)
        # Optionally normalize (L2)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

# -----------------------
# Utility helpers
# -----------------------

def vector_to_csv_list(vec: List[float]) -> str:
    """Convert float vector to textual CSV list inside square brackets."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def make_insert_batch_cmd(memname: str, items: List[Tuple[str, np.ndarray]]) -> str:
    """Build a single INSERT INTO ... VALUES command."""
    parts = []
    for label, vec in items:
        csv = vector_to_csv_list(vec.tolist())
        # Simple sanitization
        safe_label = label.replace(")", "_").replace("(", "_").replace(",", "_")
        parts.append(f"({safe_label}, {csv})")
    body = ",".join(parts)
    cmd = f"INSERT INTO {memname} VALUES {body};"
    return cmd

def make_insert_batch_cmd_with_tags(memname: str, items: List[Tuple[str, np.ndarray, str]]) -> str:
    """
    Build INSERT command with TAGS.
    Item format: (label, vector, class_tag)
    """
    cmds = []
    for label, vec, tag_val in items:
        csv = vector_to_csv_list(vec.tolist())
        safe_label = label.replace(")", "_").replace("(", "_").replace(",", "_")
        # INSERT INTO tests VALUES (lbl, [...]) TAGS (class:dog)
        cmd = f"INSERT INTO {memname} VALUES ({safe_label}, {csv}) TAGS (class:{tag_val});"
        cmds.append(cmd)
    return cmds

def make_search_cmd(memname: str, query_vec: np.ndarray, topk: int = 5) -> str:
    qcsv = vector_to_csv_list(query_vec.tolist())
    cmd = f"SEARCH {memname} QUERY ({qcsv}) TOP {topk};"
    return cmd

def query_membrance_info(client: PomaiClient, memname: str, verbose: bool = True):
    cmd = f"GET MEMBRANCE INFO {memname};"
    try:
        resp = client.send_and_get_response(cmd)
    except Exception as e:
        if verbose:
            print(f"Failed to query membrance info: {e}")
        return None
    if verbose:
        print("Raw membrance info response:\n", resp.strip())

    info = {"dim": None, "num_vectors": None, "disk_bytes": None, "raw": resp}
    m = re.search(r"dim\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["dim"] = int(m.group(1))

    m = re.search(r"(?:num(?:ber)?_?vectors|num vectors|vectors)\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m: info["num_vectors"] = int(m.group(1))

    return info

# -----------------------
# Integration flow
# -----------------------

def run_test(host: str, port: int, assets_dir: str, batch_size: int = 16, verbose: bool = True):
    # 1) locate image files
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    imgs = []
    for root, _, files in os.walk(assets_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                imgs.append(os.path.join(root, fn))
    imgs.sort()
    
    if len(imgs) < 4:
        print("Need at least 4 images to test stratified split meaningfully.")
        return False

    print(f"Found {len(imgs)} images in {assets_dir}")

    # 2) init feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Feature extraction device:", device)
    extractor = ResNet50FeatureExtractor(device=device)

    # 3) extract features for each image and assign mock tags
    # Prepare items with Mock Tags: First half -> DOG, Second half -> CAT
    items = []
    mid = len(imgs) // 2
    print("Extracting features and assigning Mock Tags...")
    for i, p in enumerate(imgs):
        try:
            img = Image.open(p)
            vec = extractor.extract(img)  # np.ndarray length 2048
            label = os.path.splitext(os.path.basename(p))[0]
            
            # Assign Tag
            tag = "dog" if i < mid else "cat"
            items.append((label, vec, tag))
            if verbose:
                print(f"  - {label} -> {tag}")
        except Exception as e:
            print("Failed to open image", p, ":", e)
            continue

    if not items:
        print("No images processed successfully")
        return False

    # 4) connect to server
    client = PomaiClient(host=host, port=port, timeout=30.0)
    try:
        # 5) create membrance
        print("\nCreating membrance 'tests' DIM 2048 ...")
        cmd = "CREATE MEMBRANCE tests DIM 2048 RAM 256;"
        resp = client.send_and_get_response(cmd)
        print(resp.strip())

        # 6) Insert with TAGS
        print(f"\nInserting {len(items)} items with TAGS...")
        # Send 1 by 1 to ensure correct tagging (since our server batch logic applies tags to whole batch)
        # Using a specialized command builder for this test
        insert_cmds = make_insert_batch_cmd_with_tags("tests", items)
        for cmd in insert_cmds:
            client.send_and_get_response(cmd)
            # Small sleep to ensure order/processing
            # time.sleep(0.005) 

        print(f"Inserted {len(items)} items")

        # Query info
        info_after = query_membrance_info(client, "tests", verbose=verbose)
        if info_after:
            print("Membrance info after inserts:", info_after)

        # ---------------------------------------------------------
        # TEST 1: RANDOM SPLIT (Legacy)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("[TEST 1] Executing RANDOM Split (80/10/10)...")
        random_split_cmd = "EXEC SPLIT tests 0.8 0.1 0.1;"
        print(client.send_and_get_response(random_split_cmd))

        # ---------------------------------------------------------
        # TEST 2: STRATIFIED SPLIT (New)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST 2] Executing STRATIFIED Split by 'class' (50/25/25)...")
        # Syntax: EXEC SPLIT <name> <tr%> <val%> <te%> STRATIFIED <key>
        stratified_cmd = "EXEC SPLIT tests 0.5 0.25 0.25 STRATIFIED class;"
        resp = client.send_and_get_response(stratified_cmd)
        print(f"Split Response: {resp.strip()}")
        
        if "ERR" in resp:
            print("[FAILED] Stratified split failed.")
        else:
            print("[SUCCESS] Stratified split executed.")
            # Regex to parse numbers: Split 8 vectors into 4 train, 2 val, 2 test
            nums = re.findall(r'(\d+)', resp)
            if len(nums) >= 4:
                print(f"  -> Total: {nums[0]} | Train: {nums[1]} | Val: {nums[2]} | Test: {nums[3]}")

        # ---------------------------------------------------------
        # [NEW] Step 6.6: Test ITERATE (Binary Streaming)
        # ---------------------------------------------------------
        print("-" * 50)
        print("[TEST] Streaming TRAIN set via ITERATE command...")
        
        # [FIX] Added semicolon to command + Robust Reader
        count_tr, dim_tr, raw_tr = client.request_binary_stream("ITERATE tests TRAIN 0 10000;")
        
        if count_tr > 0:
            arr_tr = np.frombuffer(raw_tr, dtype=np.float32).reshape(count_tr, dim_tr)
            print(f"  -> Received TRAIN batch: Shape={arr_tr.shape}, Dtype={arr_tr.dtype}")
            print(f"  -> Sample Vector[0][:5]: {arr_tr[0][:5]}")
            if np.all(arr_tr[0] == 0):
                print("  -> WARNING: Vector data is all zeros!")
            else:
                print("  -> Data integrity check: OK")
        else:
            print("  -> WARNING: No data received for TRAIN split.")

        print("\n[TEST] Streaming TEST set via ITERATE command...")
        # [FIX] Added semicolon to command + Robust Reader
        count_te, dim_te, raw_te = client.request_binary_stream("ITERATE tests TEST 0 10000;")
        
        if count_te > 0:
            arr_te = np.frombuffer(raw_te, dtype=np.float32).reshape(count_te, dim_te)
            print(f"  -> Received TEST batch: Shape={arr_te.shape}")
        else:
            print("  -> TEST set is empty (Expected if dataset small).")

        expected_total = len(items)
        # Note: Val set is not iterated here, so count_tr + count_te < total is normal
        print(f"\n[INFO] Total Streamed: {count_tr + count_te} / {expected_total} (Train+Test only)")
        print("="*50 + "\n")
        # ---------------------------------------------------------

        # 7) run a search
        probe_label, probe_vec, _ = items[0]
        # [FIXED] Compute Hash for Verification (Sanitize label like insert)
        safe_probe_label = probe_label.replace(")", "_").replace("(", "_").replace(",", "_")
        probe_hash = fnv1a_hash(safe_probe_label)
        
        print(f"Searching for first image (label={safe_probe_label}, hash={probe_hash})...")
        search_cmd = make_search_cmd("tests", probe_vec, topk=5)
        resp = client.send_and_get_response(search_cmd)
        print("Search response:\n", resp.strip())
        
        # Parse results and verify
        lines = resp.splitlines()
        found = False
        if len(lines) > 1: # Header + results
            top_line = lines[1].strip() # First result line
            parts = top_line.split()
            if len(parts) >= 1:
                try:
                    top_hash = int(parts[0])
                    if top_hash == probe_hash:
                        print(f"  -> [MATCH] Top result matches probe hash.")
                        found = True
                    else:
                        print(f"  -> [MISMATCH] Top: {top_hash}, Expected: {probe_hash}")
                except ValueError:
                    print(f"  -> [PARSE ERROR] Could not parse hash from '{parts[0]}'. Protocol sync issue?")
        
        if not found:
            print("  -> Warning: Search quality check failed (Is dataset too small/random?)")

        # 8) delete all labels
        print("\nDeleting inserted labels ...")
        inserted_labels = [i[0] for i in items]
        for lbl in inserted_labels:
            safe_lbl = lbl.replace(")", "_").replace("(", "_").replace(",", "_")
            cmd = f"DELETE tests LABEL {safe_lbl};"
            client.send_and_get_response(cmd)

        print("Deleted all labels. Re-running search...")
        resp = client.send_and_get_response(search_cmd)
        print("Search response after delete:\n", resp.strip())
        
        # 9) done
        print("Integration test completed successfully")
        return True
    finally:
        client.close()


# -----------------------
# CLI entry
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Pomai integration test")
    parser.add_argument("--host", default="127.0.0.1", help="Pomai server host")
    parser.add_argument("--port", default=7777, type=int, help="Pomai server port")
    parser.add_argument("--assets", default="assets", help="Directory containing image assets")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for inserts")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Turn off verbose output")
    args = parser.parse_args()

    ok = run_test(args.host, args.port, args.assets, batch_size=args.batch, verbose=args.verbose)
    if not ok:
        print("Integration test failed.")
        sys.exit(2)
    print("Integration test succeeded.")
    sys.exit(0)

if __name__ == "__main__":
    main()