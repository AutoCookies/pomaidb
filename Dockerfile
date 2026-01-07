# Stage 1: Build
FROM golang:1.25-alpine AS builder

WORKDIR /app

# Copy dependency files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary (Static binary)
RUN CGO_ENABLED=0 GOOS=linux go build -o pomai-server ./cmd/server/main.go

# Stage 2: Run (Dùng Alpine cho nhẹ)
FROM alpine:latest

WORKDIR /root/

# Copy binary từ builder
COPY --from=builder /app/pomai-server .

# Tạo thư mục data để mount volume
RUN mkdir -p ./data

# Thiết lập biến môi trường mặc định
ENV PORT=8080
ENV DATA_DIR=./data
ENV SHARD_COUNT=256
ENV PERSISTENCE=file

# Expose port
EXPOSE 8080

# Chạy server
CMD ["./pomai-server"]