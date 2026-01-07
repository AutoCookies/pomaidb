# Top-level Makefile (updated)
# - Builds engine_test, pomai-server, and bench target
# - Links all relevant core/ memory/ ai translation units so globals (e.g., pomai::config::runtime) are defined
# - Designed for g++ as requested (CXX variable)

# Compiler: Dùng g++
CXX ?= g++

# Flags tối ưu hóa (CỰC KỲ QUAN TRỌNG VỚI CACHE SYSTEM)
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -pthread -I.

# Thư mục đích
BUILD_DIR ?= build

# Discover common source files used by server/tests/bench
SRCS_CORE := $(wildcard core/*.cc)
SRCS_MEM  := $(wildcard memory/*.cc)
SRCS_AI   := $(wildcard ai/*.cc)
SRCS_COMMON := $(SRCS_CORE) $(SRCS_MEM) $(SRCS_AI)

# Individual entry points
SRC_ENGINE := tests/engine_check.cc
SRC_MAIN   := main.cc
SRC_BENCH  := benchmarks/pwp_bench.cc
SRC_TEST_VEC := tests/test_vector_index.cc

# Targets
ENGINE_BIN := engine_test
SERVER_BIN := pomai-server
BENCH_BIN  := $(BUILD_DIR)/pwp_bench
TEST_VEC_BIN := $(BUILD_DIR)/test_vector_index

.PHONY: all clean run bench test_vector print-sources

all: $(ENGINE_BIN) $(SERVER_BIN)

# Engine test (links in all common sources so global variables are satisfied)
$(ENGINE_BIN): $(SRC_ENGINE) $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_ENGINE) $(SRCS_COMMON)

# Server binary (example main)
$(SERVER_BIN): $(SRC_MAIN) $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_MAIN) $(SRCS_COMMON)

# Bench target: builds bench binary into build/
$(BENCH_BIN): $(SRC_BENCH) $(SRCS_COMMON)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_BENCH) $(SRCS_COMMON)

# Vector test target (build + run)
$(TEST_VEC_BIN): $(SRC_TEST_VEC) $(SRCS_COMMON)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_TEST_VEC) $(SRCS_COMMON)

test_vector: $(TEST_VEC_BIN)
	@echo "Running vector index unit test"
	@./$(TEST_VEC_BIN)

# bench helper
bench: $(BENCH_BIN)
	@echo "Running bench binary: $(BENCH_BIN)"
	@$(BENCH_BIN)

# convenience run for engine_test
run: $(ENGINE_BIN)
	@echo "--- [POMAI KERNEL START] ---"
	@./$(ENGINE_BIN)
	@echo "--- [POMAI KERNEL END] ---"

clean:
	-rm -f $(ENGINE_BIN) $(SERVER_BIN)
	-rm -rf $(BUILD_DIR)
	-find . -name '*.o' -delete

# Show which sources will be built (debug)
print-sources:
	@echo "SRCS_COMMON ="
	@for f in $(SRCS_COMMON); do echo "  $$f"; done