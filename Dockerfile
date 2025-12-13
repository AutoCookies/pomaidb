# File: Dockerfile
FROM golang:1.25-alpine AS builder

WORKDIR /app

# Copy all source
COPY . . 

# Download deps and build from package path
RUN go mod download && \
    CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o cache-server \
    ./cmd/server

# Final stage
FROM alpine:latest

RUN apk --no-cache add wget ca-certificates

WORKDIR /app

COPY --from=builder /app/cache-server . 

RUN mkdir -p /data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    CMD wget -q --spider http://localhost:8080/health || exit 1

CMD ["./cache-server"]