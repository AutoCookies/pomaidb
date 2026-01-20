#!/usr/bin/env python3
"""
ingest_and_check_simple_with_train.py

Extends the original 'ingest_and_check_simple.py' to include a
Training Loop (Autoencoder) using the ITERATE TRAIN/TEST protocol.

Usage:
  python3 ingest_and_check_simple_with_train.py --mem cifar10_features --do-train
"""

import argparse
import socket
import time
import sys
import os
from typing import Optional, Tuple

try:
    import numpy as np
except Exception:
    print("Install numpy: pip install numpy")
    raise

# optional PyTorch feature extraction
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH = True
except Exception:
    TORCH = False

# -----------------------
# Minimal Pomai client
# -----------------------
class PomaiClient:
    def __init__(self, host='127.0.0.1', port=7777, timeout=60.0): # Tăng timeout cho việc train
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def send(self, cmd: str):
        if not cmd.endswith("\n"):
            cmd = cmd + "\n"
        self.sock.sendall(cmd.encode("utf-8"))

    def recv_until_marker(self, marker: bytes = b"<END>\n") -> bytes:
        buf = b""
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if marker in buf:
                break
        return buf

    def send_and_get_text(self, cmd: str) -> str:
        self.send(cmd)
        raw = self.recv_until_marker()
        try:
            s = raw.decode("utf-8", errors="replace")
        except Exception:
            s = ""
        idx = s.find("<END>")
        return s[:idx] if idx != -1 else s

    def request_binary_stream_header(self, cmd: str) -> Tuple[Optional[str], int, int, int]:
        """
        Send command, read and parse header line only.
        Returns (dtype_token_or_None, count, dim, total_bytes).
        Caller must consume the body (total_bytes) afterwards to keep protocol in sync.
        """
        self.send(cmd)
        hdr = b""
        while b"\n" not in hdr:
            ch = self.sock.recv(1)
            if not ch:
                raise ConnectionError("socket closed while reading header")
            hdr += ch
        header_line = hdr.decode("utf-8", errors="replace").strip()
        parts = header_line.split()
        
        # Accept both "OK BINARY" and "OK BINARY_PAIR"
        if not (header_line.startswith("OK BINARY")):
             # Nếu server trả về lỗi, drain và raise
             self.recv_until_marker()
             # Raise error nhẹ nhàng hơn để code phía sau xử lý
             return None, 0, 0, 0
             
        # detect optional dtype token at parts[2]
        dtype_token = None
        idx = 2
        known = {"float32","float64","int32","int8","float16","fp16","double"}
        if len(parts) >= 3 and parts[2].lower() in known:
            dtype_token = parts[2].lower()
            idx = 3
        
        try:
            count = int(parts[idx])
            dim = int(parts[idx+1])
            total_bytes = int(parts[idx+2])
        except:
             self.recv_until_marker()
             return None, 0, 0, 0
             
        return dtype_token, count, dim, total_bytes

    def drain_bytes(self, n: int):
        """Read and discard exactly n bytes (or until EOF)."""
        remaining = n
        while remaining > 0:
            chunk = self.sock.recv(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        # consume trailing protocol marker
        self.recv_until_marker()

    # --- [BỔ SUNG] Hàm đọc dữ liệu thật để Train ---
    def read_payload(self, n: int) -> bytes:
        """Read exactly n bytes and return them, consuming the marker."""
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(min(65536, n - len(buf)))
            if not chunk:
                break
            buf += chunk
        # consume trailing protocol marker
        self.recv_until_marker()
        return buf

# -----------------------
# Small feature extractor (ResNet18) helper
# -----------------------
def build_resnet18():
    # try the new weights API, fall back if needed
    try:
        weights = ResNet18_Weights.DEFAULT
        m = resnet18(weights=weights)
        m.fc = nn.Identity()
        transform = weights.transforms()
    except Exception:
        m = resnet18(pretrained=True)
        m.fc = nn.Identity()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    m.eval()
    return m, transform

def extract_image_feature(path: str, model, transform, device='cpu'):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(t).reshape(-1).cpu().numpy().astype(np.float32)
    n = np.linalg.norm(feat)
    if n > 0:
        feat = feat / n
    return feat

# -----------------------
# [BỔ SUNG] Training Logic
# -----------------------
def train_autoencoder(args, dim):
    if not TORCH:
        print("PyTorch not available, skipping training.")
        return

    print("\n" + "="*50)
    print("STARTING AUTOENCODER TRAINING")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model đơn giản: Input -> Compress -> Output
    model = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Linear(128, dim)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    chunk_size = 2048
    batch_size = 128

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        train_count = 0
        
        client = PomaiClient(host=args.host, port=args.port)
        offset = 0
        
        try:
            while True:
                # Gọi ITERATE TRAIN (chỉ lấy vector, ko lấy ID)
                cmd = f"ITERATE {args.mem} TRAIN {offset} {chunk_size};"
                dtype, count, dim_ret, n_bytes = client.request_binary_stream_header(cmd)

                if count == 0: break
                
                payload = client.read_payload(n_bytes)
                
                # Convert bytes -> float32 numpy -> torch tensor
                # Lưu ý: Server trả về float32 raw
                floats = np.frombuffer(payload, dtype=np.float32).copy()
                if floats.size != count * dim_ret:
                    print("WARN: Size mismatch in training stream")
                    break

                full_batch = floats.reshape(count, dim_ret)
                
                # Mini-batch loop
                n_batches = (count + batch_size - 1) // batch_size
                for i in range(n_batches):
                    s = i * batch_size
                    e = min(s + batch_size, count)
                    
                    data = torch.tensor(full_batch[s:e]).to(device)
                    
                    # Sanity check
                    if torch.isnan(data).any(): continue

                    optimizer.zero_grad()
                    recon = model(data)
                    loss = criterion(recon, data)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_count += 1
                
                offset += count
                if count < chunk_size: break
        except Exception as e:
            print(f"Train Error: {e}")
        finally:
            client.close()

        if train_count > 0:
            print(f" -> Train Loss: {train_loss/train_count:.6f}")
        else:
            print(" -> Train: No data.")

        # --- TEST PHASE ---
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        client = PomaiClient(host=args.host, port=args.port)
        offset = 0
        
        try:
            while True:
                # Gọi ITERATE TEST
                cmd = f"ITERATE {args.mem} TEST {offset} {chunk_size};"
                dtype, count, dim_ret, n_bytes = client.request_binary_stream_header(cmd)

                if count == 0: break
                
                payload = client.read_payload(n_bytes)
                floats = np.frombuffer(payload, dtype=np.float32).copy()
                full_batch = floats.reshape(count, dim_ret)
                
                with torch.no_grad():
                    data = torch.tensor(full_batch).to(device)
                    recon = model(data)
                    loss = criterion(recon, data)
                    test_loss += loss.item()
                    test_count += 1
                
                offset += count
                if count < chunk_size: break
        except Exception as e:
             print(f"Test Error: {e}")
        finally:
            client.close()
            
        if test_count > 0:
            print(f" -> Test Loss : {test_loss/test_count:.6f}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7777, type=int)
    parser.add_argument("--mem", default="simple_check")
    parser.add_argument("--num", default=8, type=int)
    parser.add_argument("--assets", default="", help="image folder (optional)")
    parser.add_argument("--use-random", action="store_true", help="use random vectors instead of images")
    # [BỔ SUNG] Tham số cho việc train
    parser.add_argument("--do-train", action="store_true", help="Run autoencoder training after ingest")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    args = parser.parse_args()

    # prepare vectors
    vectors = []
    # (Logic cũ: chuẩn bị vector)
    if not args.use_random and TORCH and args.assets and os.path.isdir(args.assets):
        model, transform = build_resnet18()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        files = [os.path.join(args.assets, fn) for fn in sorted(os.listdir(args.assets)) if fn.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
        files = files[:args.num]
        for p in files:
            try:
                v = extract_image_feature(p, model, transform, device)
                vectors.append(v)
            except Exception as e:
                print("skip", p, e)
    
    # Nếu không ingest ảnh, tạo random (hoặc nếu user muốn train trên DB có sẵn, vector list này sẽ dùng để tạo DB ảo nếu cần)
    # Tuy nhiên, nếu user chỉ định DB có sẵn (ví dụ cifar10_features), ta nên cẩn thận việc Create/Insert.
    # Logic script này mặc định là tạo DB mới.
    if not vectors and args.num > 0:
        dim = 512
        for i in range(args.num):
            v = np.random.randn(dim).astype(np.float32)
            v = v / max(1e-12, np.linalg.norm(v))
            vectors.append(v)
    
    # Nếu không có vector nào (ví dụ user muốn train trên DB cũ mà không insert thêm),
    # Ta cần lấy dim từ DB. Ở đây giả sử dim 512 hoặc user phải đưa ít nhất 1 vector.
    dim = 512
    if vectors:
        dim = vectors[0].shape[0]

    print(f"Target Membrance: '{args.mem}' (Dim={dim})")

    client = PomaiClient(host=args.host, port=args.port, timeout=20.0)
    try:
        # 1. Ingest Data (Chỉ làm nếu có vectors)
        if vectors:
            # create membrance (Ignore error if exists)
            resp = client.send_and_get_text(f"CREATE MEMBRANCE {args.mem} DIM {dim} DATA_TYPE float32 RAM 256;")
            print("CREATE RESP:", resp.strip())

            # insert
            tuples = []
            for i, v in enumerate(vectors):
                csv = ",".join(f"{float(x):.6f}" for x in v.tolist())
                tuples.append(f"({i}, [{csv}])")
            
            # Chia batch nhỏ để insert nếu quá nhiều
            batch_size = 50
            for i in range(0, len(tuples), batch_size):
                chunk = tuples[i:i+batch_size]
                sql = f"INSERT INTO {args.mem} VALUES " + ",".join(chunk) + ";"
                r = client.send_and_get_text(sql)
                print(f"INSERT BATCH {i}: {r.strip()}")

            time.sleep(0.8)

            # Split
            try:
                s = client.send_and_get_text(f"EXEC SPLIT {args.mem} 0.8 0.1 0.1;")
                print("SPLIT RESP:", s.strip())
                time.sleep(0.5)
            except Exception as e:
                print("EXEC SPLIT failed/ignored:", e)
        
        # 2. [BỔ SUNG] Chạy Training Model
        if args.do_train:
            # Đóng kết nối cũ để thread sạch sẽ
            client.close()
            # Gọi hàm train
            train_autoencoder(args, dim)
            # Mở lại kết nối để cleanup
            client = PomaiClient(host=args.host, port=args.port, timeout=20.0)

    finally:
        # cleanup (Chỉ drop nếu là test db tạm thời, nếu user muốn giữ lại thì bỏ dòng này)
        if args.mem == "simple_check": # Chỉ drop db mặc định
            try:
                client.send_and_get_text(f"DROP MEMBRANCE {args.mem};")
                print(f"Dropped temporary membrance {args.mem}")
            except Exception:
                pass
        client.close()

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)