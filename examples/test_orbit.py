import socket
import struct
import random
import time

def read_n_responses(sock, n):
    """Đọc đúng n phản hồi từ server"""
    responses = []
    buffer = b""
    while len(responses) < n:
        try:
            chunk = sock.recv(4096)
            if not chunk: break
            buffer += chunk
            while b"\r\n" in buffer:
                line, buffer = buffer.split(b"\r\n", 1)
                responses.append(line)
                if len(responses) == n: break
        except BlockingIOError:
            continue
    return responses

def test_orbit_robust():
    host = '127.0.0.1'
    port = 7777    
    print(f"[Client] Connecting to Pomai Orbit...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((host, port))

    dim = 128
    num_vectors = 10000 # Tăng lên 10k để test độ ổn định
    batch_size = 100    # Gửi mỗi lần 100 lệnh để tránh Deadlock
    
    # 0. WARM UP
    print("[Client] Sending Warm-up vector...")
    warm_vec = struct.pack(f'{dim}f', *[0.1]*dim)
    buf = f"*3\r\n$4\r\nVSET\r\n$4\r\nwarm\r\n${len(warm_vec)}\r\n".encode() + warm_vec + b"\r\n"
    s.sendall(buf)
    s.recv(4096)
    print("[Client] Server ready.")

    # 1. Batched Insert
    print(f"[Client] Inserting {num_vectors} vectors (Batch Size: {batch_size})...")
    
    t0 = time.time()
    
    for i in range(0, num_vectors, batch_size):
        pipeline_buf = b""
        current_batch = 0
        
        # Build Batch
        for j in range(i, min(i + batch_size, num_vectors)):
            key = f"vec:{j}".encode()
            vec = struct.pack(f'{dim}f', *[random.random() for _ in range(dim)])
            cmd = f"*3\r\n$4\r\nVSET\r\n${len(key)}\r\n".encode() + key + b"\r\n" + \
                  f"${len(vec)}\r\n".encode() + vec + b"\r\n"
            pipeline_buf += cmd
            current_batch += 1
            
        # Send Batch
        s.sendall(pipeline_buf)
        
        # Read Batch Responses (QUAN TRỌNG: Giải phóng buffer ngay)
        resps = read_n_responses(s, current_batch)
        
        # In tiến độ
        if (i + batch_size) % 1000 == 0:
            print(f"   -> Inserted {i + batch_size}/{num_vectors}...")

    dt = time.time() - t0
    qps = num_vectors / dt
    print(f"[Result] Insert Time: {dt:.4f}s | QPS: {qps:.0f} vectors/sec")

    # 2. Search
    print("[Client] Searching...")
    query_vec = struct.pack(f'{dim}f', *[random.random() for _ in range(dim)])
    cmd = f"*3\r\n$7\r\nVSEARCH\r\n${len(query_vec)}\r\n".encode() + query_vec + b"\r\n$2\r\n10\r\n"
    
    t0 = time.time()
    s.sendall(cmd)
    resp = s.recv(4096)
    dt = (time.time() - t0) * 1000
    print(f"[Result] Search Latency: {dt:.2f} ms")
    # print(f"[Client] Response:\n{resp.decode(errors='ignore')}")
    
    s.close()

if __name__ == "__main__":
    test_orbit_robust()