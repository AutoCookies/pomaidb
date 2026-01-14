FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src ./src

RUN mkdir -p build

RUN g++ -O3 -ffast-math -pthread \
    src/main.cc \
    src/core/*.cc \
    src/ai/*.cc \
    src/memory/*.cc \
    src/utils/*.cc \
    src/external/*.cc \
    src/facade/*.cc \
    src/tools/*.cc \
    -o build/pomai_server

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash pomai
USER pomai

WORKDIR /app

COPY --from=builder /app/build/pomai_server .

USER root
RUN mkdir -p /data && chown pomai:pomai /data
USER pomai

ENV POMAI_DB_DIR=/data
ENV POMAI_PORT=7777

EXPOSE 7777

CMD ["./pomai_server"]