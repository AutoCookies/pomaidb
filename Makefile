# Top-level Makefile - wrapper around build.sh
.PHONY: all clean lib server cli

BUILD_SCRIPT := ./build.sh
CLI_SRC := src/pomai_cli.cc
CLI_BIN := build/pomai_cli
LIB := build/libsrc.a

all: server cli

lib:
	$(BUILD_SCRIPT)

server: lib
	@echo "Server built (if src/main.cc or examples/main.cc present) at build/pomai_server"

cli: lib
	@if [ -f $(CLI_SRC) ]; then \
		echo "Building Pomai CLI at $(CLI_BIN)"; \
		g++ -std=c++17 -O2 -g -I. -Isrc -pthread -Wall -Wextra $(CLI_SRC) $(LIB) -o $(CLI_BIN); \
		echo "Pomai CLI built at $(CLI_BIN)"; \
	else \
		echo "No CLI source $(CLI_SRC) found; skipping CLI build."; \
	fi

clean:
	$(BUILD_SCRIPT) clean
	@rm -f $(CLI_BIN)