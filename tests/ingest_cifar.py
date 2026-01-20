import torch
import torchvision
import torchvision.transforms as transforms
import socket
import numpy as np
from tqdm import tqdm
import sys

# --- CẤU HÌNH ---
HOST = '127.0.0.1'
PORT = 7777
DB_NAME = 'cifar10_features'
DIM = 512  # ResNet18 output dim
BATCH_SIZE = 128

def send_cmd(sock, cmd):
    if not cmd.endswith('\n'): cmd += '\n'
    sock.sendall(cmd.encode('utf-8'))
    # Đọc phản hồi đơn giản (OK/ERR)
    data = sock.recv(4096)
    return data.decode('utf-8').strip()

def setup_pomai():
    print(f"Connecting to PomaiDB {HOST}:{PORT}...")
    s = socket.create_connection((HOST, PORT))
    
    # 1. Xóa cũ tạo mới
    send_cmd(s, f"DROP MEMBRANCE {DB_NAME};")
    print(f"Creating Membrance {DB_NAME} (Dim: {DIM})...")
    resp = send_cmd(s, f"CREATE MEMBRANCE {DB_NAME} DIM {DIM} DATA_TYPE float32 RAM 1024;")
    print(f"Server: {resp}")
    return s

def extract_and_ingest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Model: ResNet18 (Pretrained) - Bỏ lớp FC cuối
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity() # Thay lớp cuối bằng Identity để lấy raw features
    model.to(device)
    model.eval()

    # 2. Data: CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(224), # ResNet cần ảnh to hơn 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    s = setup_pomai()
    
    print("Starting Ingestion...")
    total_vectors = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device)
            
            # Extract Features
            features = model(inputs) # [Batch, 512]
            
            # Normalize vector (L2 norm) - Quan trọng cho Vector Search & Training ổn định
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            features_np = features.cpu().numpy()
            labels_np = targets.numpy() # Đây là class ID (0-9) của ảnh, ta dùng làm Label vector luôn để tiện check
            
            # Build SQL Batch
            # Cấu trúc: INSERT INTO cifar10 VALUES (id, [vec]) TAGS (class=X);
            # Tuy nhiên để đơn giản cho training loop sau này, ta sẽ dùng ID là một số index tăng dần
            # và lưu class thật vào một nơi khác hoặc giả lập ID = (index << 4) | class (mẹo bitwise)
            # Ở đây tôi dùng cách đơn giản: Lưu ID = index toàn cục.
            
            sql_parts = []
            # Trong ingest_cifar.py
            for i in range(len(features_np)):
                vec_str = ",".join([f"{x:.4f}" for x in features_np[i]])
                # SỬA: ID chỉ là index thuần túy từ 0 -> 49999
                global_idx = total_vectors + i 
                sql_parts.append(f"({global_idx}, [{vec_str}])")
            
            sql = f"INSERT INTO {DB_NAME} VALUES {','.join(sql_parts)};"
            
            # Send to Pomai
            send_cmd(s, sql)
            total_vectors += len(features_np)

    print(f"\n[DONE] Ingested {total_vectors} vectors into PomaiDB.")
    
    # Checkpoint để đảm bảo an toàn
    print("Triggering Checkpoint...")
    print(send_cmd(s, "EXEC CHECKPOINT;"))
    s.close()

if __name__ == "__main__":
    extract_and_ingest()