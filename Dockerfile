# --- STAGE 1: Build ---
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy dependency definition
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build -o server main.go

# --- STAGE 2: Runtime ---
FROM alpine:latest

WORKDIR /root/

# Copy binary từ stage builder
COPY --from=builder /app/server .

# Port mặc định trong code Go của bạn là 8080
EXPOSE 8080

CMD ["./server"]