import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 7777
DB_NAME = 'cifar10_features'
DIM = 512
BATCH_SIZE = 128    # Batch size cho GPU
CHUNK_SIZE = 1024   # Kéo 1 lần 1024 vector về RAM
EPOCHS = 5
LR = 0.001

# Logic chia tập thủ công (Client-side Split)
TOTAL_VECTORS = 50000
TRAIN_LIMIT   = 40000  # 0 -> 39999
VAL_LIMIT     = 45000  # 40000 -> 44999
TEST_LIMIT    = 50000  # 45000 -> 49999

# -------------------------------------------------------------------
# NETWORK CLIENT (Mượn từ test_pomai_integration.py - Chuẩn 100%)
# -------------------------------------------------------------------
class PomaiClient:
    def __init__(self, host, port, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)

    def close(self):
        try: self.sock.close()
        except: pass

    def send(self, cmd: str):
        # Luôn thêm \n và ; để server cắt lệnh chuẩn xác
        if not cmd.endswith("\n"): cmd = cmd + "\n"
        self.sock.sendall(cmd.encode("utf-8"))

    def recv_header_line(self):
        # Đọc từng byte cho đến khi gặp \n (Logic của test script)
        buf = b""
        while b"\n" not in buf:
            chunk = self.sock.recv(1)
            if not chunk: raise ConnectionError("Socket closed")
            buf += chunk
        return buf.decode("utf-8").strip()

    def recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk: raise ConnectionError("Incomplete read")
            buf += chunk
        return buf

# -------------------------------------------------------------------
# STREAMER
# -------------------------------------------------------------------
class PomaiStreamer:
    def stream_range(self, start_idx, end_idx):
        """
        Duyệt ID từ start_idx đến end_idx. 
        Dùng lệnh ITERATE thuần túy: ITERATE <db> PAIR <off> <lim>;
        """
        client = PomaiClient(HOST, PORT)
        
        curr = start_idx
        try:
            while curr < end_idx:
                # Tính lượng cần lấy
                want = min(CHUNK_SIZE, end_idx - curr)
                
                # Lệnh SQL chuẩn (Có chấm phẩy)
                cmd = f"ITERATE {DB_NAME} PAIR {curr} {want};"
                client.send(cmd)

                # 1. Đọc Header
                header = client.recv_header_line()
                
                # Xử lý các trường hợp server trả về
                if not header.startswith("OK BINARY"):
                    # Nếu server trả về lỗi hoặc text lạ
                    if "ERR" in header: print(f"[Server ERR] {header}")
                    break
                
                # Parse: OK BINARY_PAIR <dtype> <count> <dim> <bytes>
                parts = header.split()
                try:
                    count = int(parts[-3])
                    total_bytes = int(parts[-1])
                except:
                    print(f"[Client] Bad header: {header}")
                    break

                if count == 0: break

                # 2. Đọc Payload
                payload = client.recv_exact(total_bytes)

                # 3. Parse Binary (ID + Vector)
                # Kích thước 1 record: 8 byte ID + 512*4 byte Float
                rec_size = 8 + (DIM * 4)
                
                chunk_inputs = []
                chunk_labels = []
                
                off = 0
                for _ in range(count):
                    # Lấy ID
                    rid = struct.unpack_from('<Q', payload, off)[0]
                    # [HACK] Lấy Label từ ID (ID = idx*100 + label)
                    lbl = rid % 10
                    
                    # Lấy Vector
                    vec_bytes = payload[off+8 : off+rec_size]
                    # Copy để tạo mảng sạch
                    vec_np = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                    
                    chunk_inputs.append(vec_np)
                    chunk_labels.append(lbl)
                    off += rec_size

                # Yield Mini-Batches cho GPU
                full_X = np.array(chunk_inputs)
                full_Y = np.array(chunk_labels)
                
                # Cắt nhỏ ra batch_size để train mượt hơn
                n_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
                for i in range(n_batches):
                    s = i * BATCH_SIZE
                    e = min(s + BATCH_SIZE, count)
                    yield torch.tensor(full_X[s:e]), torch.tensor(full_Y[s:e], dtype=torch.long)

                curr += count

        finally:
            client.close()

# -------------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING on {device} ===")
    
    model = nn.Linear(DIM, 10).to(device)
    optimz = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    
    streamer = PomaiStreamer()
    
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        # --- TRAIN (0 -> 40000) ---
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        t0 = time.time()
        
        for X, Y in streamer.stream_range(0, TRAIN_LIMIT):
            X, Y = X.to(device), Y.to(device)
            optimz.zero_grad()
            out = model(X)
            loss = crit(out, Y)
            loss.backward()
            optimz.step()
            
            t_loss += loss.item()
            t_corr += (out.argmax(1) == Y).sum().item()
            t_total += Y.size(0)
            
        dt = time.time() - t0
        acc = 100 * t_corr / t_total if t_total else 0
        print(f" -> Train: Loss {t_loss/t_total*BATCH_SIZE:.4f} | Acc {acc:.2f}% | Time {dt:.2f}s")

        # --- VAL (40000 -> 45000) ---
        model.eval()
        v_corr, v_total = 0, 0
        with torch.no_grad():
            for X, Y in streamer.stream_range(TRAIN_LIMIT, VAL_LIMIT):
                X, Y = X.to(device), Y.to(device)
                v_corr += (model(X).argmax(1) == Y).sum().item()
                v_total += Y.size(0)
        
        v_acc = 100 * v_corr / v_total if v_total else 0
        print(f" -> Val  : Acc {v_acc:.2f}%")

if __name__ == "__main__":
    train()