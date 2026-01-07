# Top-level Makefile - wrapper around build.sh
.PHONY: all clean lib server

BUILD_SCRIPT := ./build.sh

all: server

lib:
	$(BUILD_SCRIPT)

server: lib
	@echo "Server built (if src/main.cc or examples/main.cc present) at build/pomai_server"

clean:
	$(BUILD_SCRIPT) clean