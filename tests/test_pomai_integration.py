#!/usr/bin/env python3
"""
test_pomai_integration.py

End-to-end integration test for Pomai server (text SQL-like protocol).

This version is extended to query the new "GET MEMBRANCE INFO" command
(after you've added server support) to inspect dim, number of vectors and
disk footprint for a membrance.

See top of file for original description and requirements.
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

    def send_and_get_response(self, cmd: str, expect_end_marker: bool = True) -> str:
        """
        Send a textual command and read until the server's "<END>\n" marker is seen.
        Returns the full response as a string (excluding the final marker).
        """
        self.send(cmd)
        chunks = []
        # read until marker
        marker = "<END>\n"
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
            if marker.encode("utf-8") in buf:
                break
        text = buf.decode("utf-8", errors="replace")
        # strip everything after marker
        idx = text.find(marker)
        if idx != -1:
            return text[:idx]
        return text

# -----------------------
# Feature extractor
# -----------------------

class ResNet50FeatureExtractor:
    """
    Use torchvision resnet50 pretrained to extract 2048-dim features from images:
    - We grab the output of the avgpool layer (2048 floats).
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        model = torch.hub.load('pytorch/vision:v0.14.0', 'resnet50', pretrained=True)
        # Remove final FC: keep everything up to avgpool
        modules = list(model.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*modules).to(self.device)
        self.backbone.eval()
        # transforms matching ImageNet training
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
            feat = self.backbone(t)  # shape 1 x 2048 x 1 x 1
            feat = feat.reshape(feat.shape[0], -1)  # 1 x 2048
            arr = feat.cpu().numpy().astype(np.float32).reshape(-1)
        # Optionally normalize (L2)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

# -----------------------
# Utility helpers
# -----------------------

def vector_to_csv_list(vec: List[float]) -> str:
    """
    Convert float vector to textual CSV list inside square brackets, e.g.:
      [0.123, -0.432, ...]
    We format floats with 6 decimal digits to keep commands reasonably sized.
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def make_insert_batch_cmd(memname: str, items: List[Tuple[str, np.ndarray]]) -> str:
    """
    Build a single INSERT INTO ... VALUES (...),(...); command for a list of (label, vector)
    """
    parts = []
    for label, vec in items:
        csv = vector_to_csv_list(vec.tolist())
        # label is passed as a bare token in server grammar (server expects it as token and hashes it).
        # We will pass label without quotes; if label contains spaces avoid that.
        safe_label = label.replace(")", "_").replace("(", "_").replace(",", "_")
        parts.append(f"({safe_label}, {csv})")
    body = ",".join(parts)
    cmd = f"INSERT INTO {memname} VALUES {body};"
    return cmd

def make_search_cmd(memname: str, query_vec: np.ndarray, topk: int = 5) -> str:
    qcsv = vector_to_csv_list(query_vec.tolist())
    cmd = f"SEARCH {memname} QUERY ({qcsv}) TOP {topk};"
    return cmd

# -----------------------
# Membrance info parsing
# -----------------------

def query_membrance_info(client: PomaiClient, memname: str, verbose: bool = True):
    """
    Send GET MEMBRANCE INFO <memname>; and try to parse common numeric fields.
    Expected server response should include human-readable fields (server must be extended).
    This function is forgiving: it prints raw response and tries to extract numbers.
    """
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
    # Try heuristics to extract numbers
    # Patterns like: dim=2048 or dim: 2048 or Dim 2048
    m = re.search(r"dim\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if not m:
        m = re.search(r"\bDIM(?:ENSION)?\b\s*(\d+)", resp, re.IGNORECASE)
    if m:
        info["dim"] = int(m.group(1))

    m = re.search(r"(?:num(?:ber)?_?vectors|num vectors|vectors)\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if not m:
        m = re.search(r"\bTOTAL_VECTORS\b\s*[:=]?\s*(\d+)", resp, re.IGNORECASE)
    if m:
        info["num_vectors"] = int(m.group(1))

    # disk bytes: allow bytes or GB suffix
    m = re.search(r"(?:disk[_\s]?bytes|disk_bytes|disk)\s*[:=]\s*(\d+)", resp, re.IGNORECASE)
    if m:
        info["disk_bytes"] = int(m.group(1))
    else:
        # try GB like "disk: 0.0123 GB"
        m2 = re.search(r"disk(?:[_\s]?size|[_\s]?gb|)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*GB", resp, re.IGNORECASE)
        if m2:
            gb = float(m2.group(1))
            info["disk_bytes"] = int(gb * 1024.0 * 1024.0 * 1024.0)

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
    if not imgs:
        print("No image assets found in", assets_dir)
        return False

    print(f"Found {len(imgs)} images in {assets_dir}")

    # 2) init feature extractor (CPU). If CUDA available, user may pass env var to use GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Feature extraction device:", device)
    extractor = ResNet50FeatureExtractor(device=device)

    # 3) extract features for each image
    items = []  # list of (label, vector)
    for p in imgs:
        try:
            img = Image.open(p)
        except Exception as e:
            print("Failed to open image", p, ":", e)
            continue
        vec = extractor.extract(img)  # np.ndarray length 2048
        label = os.path.splitext(os.path.basename(p))[0]
        items.append((label, vec))
        if verbose:
            print("Extracted", label, "len", vec.shape)

    if not items:
        print("No images processed successfully")
        return False

    # 4) connect to server
    client = PomaiClient(host=host, port=port, timeout=30.0)
    try:
        # 5) create membrance
        print("Creating membrance 'tests' DIM 2048 ...")
        cmd = "CREATE MEMBRANCE tests DIM 2048 RAM 256;"
        resp = client.send_and_get_response(cmd)
        print(resp.strip())

        # Query membrance info BEFORE inserts (if supported)
        info_before = query_membrance_info(client, "tests", verbose=verbose)
        if info_before:
            print("Membrance info before inserts:", info_before)

        # 6) batch insert
        print("Inserting vectors in batches (batch_size=%d) ..." % batch_size)
        inserted_labels = []
        i = 0
        total = len(items)
        while i < total:
            batch = items[i:i+batch_size]
            cmd = make_insert_batch_cmd("tests", batch)
            # send and get response
            resp = client.send_and_get_response(cmd)
            if verbose:
                print(resp.strip())
            # assume success if server responded OK (cmd prints OK)
            for label, _ in batch:
                inserted_labels.append(label)
            i += batch_size
            time.sleep(0.05)  # small pause to avoid overwhelming server

        print(f"Inserted {len(inserted_labels)} items")

        # Query membrance info AFTER inserts
        info_after = query_membrance_info(client, "tests", verbose=verbose)
        if info_after:
            print("Membrance info after inserts:", info_after)

        # 7) run a search with the first vector and validate top-1 is the same label
        probe_label, probe_vec = items[0]
        print("Searching for first image (label=%s) ..." % probe_label)
        search_cmd = make_search_cmd("tests", probe_vec, topk=5)
        resp = client.send_and_get_response(search_cmd)
        print("Search response:\n", resp.strip())
        # parse results: lines like "<label> <distance>"
        found_labels = []
        for line in resp.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("RESULTS") or line.startswith("OK") or line.startswith("ERR"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                found_labels.append(parts[0])
        if found_labels:
            print("Top result label:", found_labels[0])
            if found_labels[0] == probe_label:
                print("Search check: OK (probe found as top result)")
            else:
                print("Search check: WARNING (top result != probe). Top:", found_labels[0], "expected:", probe_label)
        else:
            print("Search returned no candidates")

        # 8) delete all labels and verify get/search no longer returns them
        print("Deleting inserted labels ...")
        for lbl in inserted_labels:
            cmd = f"DELETE tests LABEL {lbl};"
            resp = client.send_and_get_response(cmd)
            if verbose:
                print(resp.strip())

        print("Deleted all labels. Re-running search for probe to ensure it's gone.")
        resp = client.send_and_get_response(search_cmd)
        print("Search response after delete:\n", resp.strip())
        # ensure probe_label not returned
        still_found = False
        for line in resp.splitlines():
            if probe_label in line:
                still_found = True
                break
        if still_found:
            print("Delete check: FAILED - probe still returned")
            return False
        else:
            print("Delete check: OK - probe not found after delete")

        # Query membrance info AFTER deletes
        info_after_delete = query_membrance_info(client, "tests", verbose=verbose)
        if info_after_delete:
            print("Membrance info after deletes:", info_after_delete)

        # 9) done
        print("Integration test completed successfully")
        return True
    finally:
        client.close()


# -----------------------
# CLI entry
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Pomai integration test (images->features->insert->search->delete)")
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