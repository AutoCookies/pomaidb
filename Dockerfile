# --- Giai đoạn 1: Builder ---
FROM golang:1.25-alpine AS builder

# Cài đặt git để tải dependencies (nếu cần)
RUN apk add --no-cache git

WORKDIR /app

# Copy file dependency trước để tận dụng Docker Cache
COPY go.mod go.sum ./
RUN go mod download

# Copy toàn bộ code nguồn
COPY . .

# Build ứng dụng
# [QUAN TRỌNG] Trỏ đúng vào file main.go nằm trong cmd/server
RUN CGO_ENABLED=0 GOOS=linux go build -o pomai-server ./cmd/server

# --- Giai đoạn 2: Runner (Siêu nhẹ) ---
FROM alpine:latest

# Cài đặt CA Certificates để gọi HTTPS (quan trọng cho Firebase/Resend)
# Cài thêm tzdata để log đúng giờ Việt Nam
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

# Copy binary từ giai đoạn builder
COPY --from=builder /app/pomai-server .

# Tạo thư mục chứa dữ liệu cache (Persistence)
RUN mkdir -p /data/cache

# Expose port theo .env
EXPOSE 8080

# Chạy server
CMD ["./pomai-server"]