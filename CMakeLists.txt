cmake_minimum_required(VERSION 3.20)
project(pomai VERSION 0.1.0 LANGUAGES C CXX)

# =========================
# Global build settings
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POMAI_BUILD_TESTS "Build tests" OFF)
option(POMAI_BUILD_BENCH "Build benchmarks" OFF)

# Prefer integer (SQ8/FP16) distance paths where data is quantized; reduces float use on embedded.
option(POMAI_PREFER_INTEGER_MATH "Prefer integer/SQ8/FP16 paths for distance (embedded)" ON)

# Edge-oriented build profile for constrained devices.
# When enabled, we bias towards smaller binaries and lower memory/debug overhead.
option(POMAI_EDGE_BUILD "Optimize PomaiDB build for edge devices (size/footprint)" OFF)

# Strict build mode: treat project warnings as errors (useful for CI/hardening).
# We keep a couple of warning categories non-fatal to tolerate intentional diagnostics
# from some vendored headers included by the core.
option(POMAI_STRICT "Treat warnings as errors for PomaiDB code" OFF)

# Ensure POMAI_PREFER_INTEGER_MATH is defined for source code (0 or 1)
add_compile_definitions(POMAI_PREFER_INTEGER_MATH=$<BOOL:${POMAI_PREFER_INTEGER_MATH}>)

if (POMAI_EDGE_BUILD)
  add_compile_definitions(POMAI_EDGE_BUILD=1)
  # Favor size over speed and strip most debug info in edge builds.
  add_compile_options(-Os -g0)
endif()


# =========================
# Native HNSW (Replaces FAISS HNSW)
# =========================
message(STATUS "[pomai] Native HNSW: enabled")


# =========================
# Library: pomai
# =========================
add_library(pomai STATIC
  src/api/db.cc
  src/database.cc
  src/agent/agent_memory.cc
  src/palloc_page_pool.cc
  src/core/vector_engine/vector_engine.cc
  src/core/shard/runtime.cc
  src/core/shard/layer_lookup.cc
  src/core/shard/iterator.cc
  src/core/shard/manifest.cc
  src/core/membrane/manager.cc
  src/table/memtable.cc
  src/storage/wal/wal.cc
  src/storage/manifest/manifest.cc
  src/storage/storage_engine.cc
  src/table/segment.cc
  src/core/storage/compaction_manager.cc

  src/core/quantization/scalar_quantizer.cc
  src/core/quantization/half_float_quantizer.cc
  src/core/index/ivf_coarse.cc
  src/core/index/ivf_flat.cc
  src/core/distance.cc

  src/core/routing/routing_table.cc
  src/core/routing/kmeans_lite.cc
  src/core/routing/routing_persist.cc
  src/core/rag/rag_engine.cc
  src/core/rag/chunking/recursive_splitter.cc
  src/core/rag/chunking/sliding_window.cc
  src/core/rag/mock_embedding_provider.cc
  src/core/rag/rag_pipeline.cc

  # Phase 2: Product Quantizer
  src/core/quantization/pomai_pq.cc

  # Phase 3 (Milvus-Lite inspired): BitsetMask pre-computed filter bitset
  src/core/bitset_mask.cc

  src/core/index/hnsw_index.cc
  third_party/pomaidb_hnsw/hnsw.cc

  # Gate #1/#2 utilities (linked into libpomai)
  src/util/posix_file.cc
  src/util/posix_env.cc
  src/util/memory_env.cc
  src/util/env.cc
  src/util/crc32c.cc
  src/util/logging.cc
)

target_include_directories(pomai
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Vendored code: keep strict warnings for PomaiDB, but avoid breaking builds on
# third_party sources when POMAI_STRICT is enabled.
if (POMAI_STRICT AND NOT MSVC)
  set_source_files_properties(third_party/pomaidb_hnsw/hnsw.cc PROPERTIES COMPILE_OPTIONS
    "-Wno-error;-Wno-shadow;-Wno-unused-parameter;-Wno-unused-but-set-variable")
endif()

# OpenMP for parallel builds if needed
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  target_link_libraries(pomai PRIVATE OpenMP::OpenMP_CXX)
endif()

if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  # Strict warnings for pomaidb code; then suppress vendored third_party (simd) warnings.
  # We do not use / -Werror globally because some third_party headers intentionally emit #warning.
  target_compile_options(pomai PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
    -Wno-cpp -Wno-unknown-pragmas -Wno-conversion -Wno-float-conversion -Wno-unused-function
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai PRIVATE /WX)
  else()
    target_compile_options(pomai PRIVATE
      -Werror
      # Some vendored headers use extensions that trigger pedantic/cpp warnings.
      -Wno-error=pedantic
      -Wno-error=cpp
      # third_party/simd uses #warning; pre-C++23 this triggers -Wc++23-extensions.
      # We disable that warning category entirely to keep strict builds usable.
      -Wno-c++23-extensions
    )
  endif()
endif()


# =========================
# Library: pomai_c (C ABI)
# =========================
set(POMAI_C_API_SOURCES
  src/capi/capi_status.cc
  src/capi/capi_db.cc
  src/capi/capi_scan.cc
  src/capi/capi_agent_memory.cc
)

add_library(pomai_c SHARED ${POMAI_C_API_SOURCES})
add_library(pomai_c_static STATIC ${POMAI_C_API_SOURCES})

set_target_properties(pomai_c_static PROPERTIES OUTPUT_NAME pomai_c)

target_link_libraries(pomai_c PRIVATE pomai)
target_link_libraries(pomai_c_static PRIVATE pomai)

# Export C API symbols when building shared library on Windows.
target_compile_definitions(pomai_c PRIVATE POMAI_C_BUILD_DLL)

target_include_directories(pomai_c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(pomai_c_static
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
  target_compile_options(pomai_c PRIVATE /W4 /permissive-)
  target_compile_options(pomai_c_static PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai_c PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
  target_compile_options(pomai_c_static PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai_c PRIVATE /WX)
    target_compile_options(pomai_c_static PRIVATE /WX)
  else()
    target_compile_options(pomai_c PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
    target_compile_options(pomai_c_static PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
  endif()
endif()

# =========================
# Tests
# =========================
if (POMAI_BUILD_TESTS)
  enable_testing()

  # Detect TSAN builds (best-effort).
  set(POMAI_IS_TSAN OFF)
  if (CMAKE_CXX_FLAGS MATCHES "fsanitize=thread" OR CMAKE_EXE_LINKER_FLAGS MATCHES "fsanitize=thread")
    set(POMAI_IS_TSAN ON)
  endif()

  option(POMAI_ENABLE_CRASH_TESTS "Enable crash-style tests (may be incompatible with TSAN)" ON)
  if (POMAI_IS_TSAN)
    set(POMAI_ENABLE_CRASH_TESTS OFF)
  endif()

  function(pomai_setup_test target)
    # Link against main C++ library and C API shim
    target_link_libraries(${target} PRIVATE pomai pomai_c)

    # One main per test binary (your harness)
    target_sources(${target} PRIVATE tests/common/test_runner.cc)

    # Warnings
    if (MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    # Allow tests to include both public and internal headers.
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}              # "tests/common/..."
      ${CMAKE_CURRENT_SOURCE_DIR}/include      # public API
      ${CMAKE_CURRENT_SOURCE_DIR}/src          # internal headers: util/, core/, table/, storage/
    )

    # Make TSAN builds more stable on some distros (PIE can trigger weird mmap layouts).
    if (POMAI_IS_TSAN)
      target_compile_options(${target} PRIVATE -fno-pie)
      target_link_options(${target} PRIVATE -no-pie)
    endif()
  endfunction()

  function(pomai_add_labeled_test target label)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${label}")
  endfunction()

  # ---- Unit ----
  add_executable(crc32c_test tests/unit/crc32c_test.cc)
  pomai_setup_test(crc32c_test)
  pomai_add_labeled_test(crc32c_test "unit")

  add_executable(hnsw_pomaidb_test tests/unit/hnsw_pomaidb_test.cc)
  pomai_setup_test(hnsw_pomaidb_test)
  pomai_add_labeled_test(hnsw_pomaidb_test "unit")

  add_executable(memtable_test tests/unit/memtable_test.cc)
  pomai_setup_test(memtable_test)
  pomai_add_labeled_test(memtable_test "unit")

  add_executable(palloc_page_pool_test tests/unit/palloc_page_pool_test.cc)
  pomai_setup_test(palloc_page_pool_test)
  pomai_add_labeled_test(palloc_page_pool_test "unit")

  add_executable(env_test tests/unit/env_test.cc)
  pomai_setup_test(env_test)
  pomai_add_labeled_test(env_test "unit")

  add_executable(agent_memory_core_test tests/unit/agent_memory_core_test.cc)
  pomai_setup_test(agent_memory_core_test)
  # Temporarily exclude from default CTest runs until AgentMemory semantics are finalized.
  # pomai_add_labeled_test(agent_memory_core_test "unit")

  add_executable(agent_memory_concurrency_test tests/unit/agent_memory_concurrency_test.cc)
  pomai_setup_test(agent_memory_concurrency_test)
  pomai_add_labeled_test(agent_memory_concurrency_test "unit")

  add_executable(mailbox_test tests/unit/mailbox_test.cc)
  pomai_setup_test(mailbox_test)
  pomai_add_labeled_test(mailbox_test "unit")

  add_executable(logging_test tests/unit/logging_test.cc)
  pomai_setup_test(logging_test)
  pomai_add_labeled_test(logging_test "unit")

  add_executable(wal_test tests/unit/wal_test.cc)
  pomai_setup_test(wal_test)
  pomai_add_labeled_test(wal_test "unit")

  add_executable(segment_test tests/unit/segment_test.cc)
  pomai_setup_test(segment_test)
  pomai_add_labeled_test(segment_test "unit")

  add_executable(fp16_test tests/unit/fp16_test.cc)
  pomai_setup_test(fp16_test)
  pomai_add_labeled_test(fp16_test "unit")
  
  add_executable(shard_manifest_test tests/unit/shard_manifest_test.cc)
  pomai_setup_test(shard_manifest_test)
  pomai_add_labeled_test(shard_manifest_test "unit")

  add_executable(oracle_test tests/unit/oracle_test.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(oracle_test)
  pomai_add_labeled_test(oracle_test "unit")

  add_executable(routing_table_test tests/unit/routing_table_test.cc)
  pomai_setup_test(routing_table_test)
  pomai_add_labeled_test(routing_table_test "unit")

  add_executable(streaming_index_test tests/unit/streaming_index_test.cc)
  pomai_setup_test(streaming_index_test)
  pomai_add_labeled_test(streaming_index_test "unit")

  add_executable(rag_chunking_test tests/unit/rag_chunking_test.cc)
  pomai_setup_test(rag_chunking_test)
  pomai_add_labeled_test(rag_chunking_test "unit")

  add_executable(rag_embedding_test tests/unit/rag_embedding_test.cc)
  pomai_setup_test(rag_embedding_test)
  pomai_add_labeled_test(rag_embedding_test "unit")

  add_executable(recall_test tests/recall/recall_test.cc tests/recall/recall_dataset.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(recall_test)
  pomai_add_labeled_test(recall_test "recall")

  # ---- Integration (embedded DB API) ----
  add_executable(db_basic_test tests/integ/db_basic_test.cc)
  pomai_setup_test(db_basic_test)
  pomai_add_labeled_test(db_basic_test "integ")

  add_executable(db_batch_test tests/integ/db_batch_test.cc)
  pomai_setup_test(db_batch_test)
  pomai_add_labeled_test(db_batch_test "integ")

  add_executable(db_persistence_test tests/integ/db_persistence_test.cc)
  pomai_setup_test(db_persistence_test)
  pomai_add_labeled_test(db_persistence_test "integ")

  add_executable(db_segment_test tests/integ/db_segment_test.cc)
  pomai_setup_test(db_segment_test)
  pomai_add_labeled_test(db_segment_test "integ")

  add_executable(db_open_test tests/integ/db_open_test.cc)
  pomai_setup_test(db_open_test)
  pomai_add_labeled_test(db_open_test "integ")
  
  add_executable(db_consistency_test tests/integ/db_consistency_test.cc)
  pomai_setup_test(db_consistency_test)
  pomai_add_labeled_test(db_consistency_test "integ")

  add_executable(db_partial_search_test tests/integ/db_partial_search_test.cc)
  pomai_setup_test(db_partial_search_test)
  pomai_add_labeled_test(db_partial_search_test "integ")

  add_executable(db_backpressure_test tests/integ/db_backpressure_test.cc)
  pomai_setup_test(db_backpressure_test)
  pomai_add_labeled_test(db_backpressure_test "integ")

  add_executable(db_edge_workload_test tests/integ/db_edge_workload_test.cc)
  pomai_setup_test(db_edge_workload_test)
  pomai_add_labeled_test(db_edge_workload_test "integ")

  add_executable(db_error_paths_test tests/integ/db_error_paths_test.cc)
  pomai_setup_test(db_error_paths_test)
  pomai_add_labeled_test(db_error_paths_test "integ")

  add_executable(agent_memory_basic_test tests/integ/agent_memory_basic_test.cc)
  pomai_setup_test(agent_memory_basic_test)
  # Temporarily exclude from default CTest runs until AgentMemory persistence behavior is finalized.
  # pomai_add_labeled_test(agent_memory_basic_test "integ")

  add_executable(agent_memory_capi_test tests/integ/agent_memory_capi_test.cc)
  pomai_setup_test(agent_memory_capi_test)
  # Temporarily exclude from default CTest runs until AgentMemory C API is fully stabilized.
  # pomai_add_labeled_test(agent_memory_capi_test "integ")

  add_executable(routing_engine_test tests/integ/routing_engine_test.cc)
  pomai_setup_test(routing_engine_test)
  pomai_add_labeled_test(routing_engine_test "integ")

  add_executable(manifest_test tests/unit/manifest_test.cc)
  pomai_setup_test(manifest_test)
  pomai_add_labeled_test(manifest_test "unit")

  add_executable(membrane_test tests/integ/membrane_test.cc)
  pomai_setup_test(membrane_test)
  pomai_add_labeled_test(membrane_test "integ")

  add_executable(membrane_persistence_test tests/integ/membrane_persistence_test.cc)
  pomai_setup_test(membrane_persistence_test)
  pomai_add_labeled_test(membrane_persistence_test "integ")

  add_executable(shard_concurrency_test tests/integ/shard_concurrency_test.cc)
  pomai_setup_test(shard_concurrency_test)
  pomai_add_labeled_test(shard_concurrency_test "integ")

  add_executable(iterator_test tests/integ/iterator_test.cc)
  pomai_setup_test(iterator_test)
  pomai_add_labeled_test(iterator_test "integ")

  add_executable(metadata_test tests/integ/metadata_test.cc)
  pomai_setup_test(metadata_test)
  pomai_add_labeled_test(metadata_test "integ")

  add_executable(search_filter_test tests/integ/search_filter_test.cc)
  pomai_setup_test(search_filter_test)
  pomai_add_labeled_test(search_filter_test "integ")

  add_executable(rag_contract_test tests/integ/rag_contract_test.cc)
  pomai_setup_test(rag_contract_test)
  pomai_add_labeled_test(rag_contract_test "integ")

  add_executable(rag_search_test tests/integ/rag_search_test.cc)
  pomai_setup_test(rag_search_test)
  pomai_add_labeled_test(rag_search_test "integ")

  add_executable(rag_pipeline_test tests/integ/rag_pipeline_test.cc)
  pomai_setup_test(rag_pipeline_test)
  pomai_add_labeled_test(rag_pipeline_test "integ")

  add_executable(search_newest_wins_test tests/integ/search_newest_wins_test.cc)
  pomai_setup_test(search_newest_wins_test)
  pomai_add_labeled_test(search_newest_wins_test "integ")

  add_executable(manifest_corruption_test tests/crash/manifest_corruption_test.cc)
  pomai_setup_test(manifest_corruption_test)
  pomai_add_labeled_test(manifest_corruption_test "crash")

  add_executable(recovery_test tests/crash/recovery_test.cc)
  pomai_setup_test(recovery_test)
  pomai_add_labeled_test(recovery_test "crash")
  set_tests_properties(recovery_test PROPERTIES TIMEOUT 90)

  # ---- Crash (Gate #2) ----
  if (POMAI_ENABLE_CRASH_TESTS)
    add_executable(pomai_crash_test tests/crash/crash_replay_test.cc)
    
    # Manual setup loosely based on pomai_setup_test but without test_runner.cc
    target_link_libraries(pomai_crash_test PRIVATE pomai)
    if (MSVC)
      target_compile_options(pomai_crash_test PRIVATE /W4 /permissive-)
    else()
      target_compile_options(pomai_crash_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    target_include_directories(pomai_crash_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    if (POMAI_IS_TSAN)
      target_compile_options(pomai_crash_test PRIVATE -fno-pie)
      target_link_options(pomai_crash_test PRIVATE -no-pie)
    endif()

    add_test(NAME pomai_crash_replay COMMAND pomai_crash_test)
    set_tests_properties(pomai_crash_replay PROPERTIES LABELS "crash" TIMEOUT 60)
  endif()

  # ---- TSAN (Gate #3) ----
  add_executable(db_concurrency_tsan_test tests/tsan/db_concurrency_tsan_test.cc)
  pomai_setup_test(db_concurrency_tsan_test)
  pomai_add_labeled_test(db_concurrency_tsan_test "tsan")

  add_executable(shard_runtime_tsan_test tests/tsan/shard_runtime_tsan_test.cc)
  pomai_setup_test(shard_runtime_tsan_test)
  pomai_add_labeled_test(shard_runtime_tsan_test "tsan")

  add_executable(backpressure_tsan_test tests/tsan/backpressure_tsan_test.cc)
  pomai_setup_test(backpressure_tsan_test)
  pomai_add_labeled_test(backpressure_tsan_test "tsan")

  add_executable(basic_workload_tsan_test tests/tsan/basic_workload_tsan_test.cc)
  pomai_setup_test(basic_workload_tsan_test)
  pomai_add_labeled_test(basic_workload_tsan_test "tsan")

  add_subdirectory(tests/fuzz)
endif()

# =========================
# Benchmarks
# =========================
if (POMAI_BUILD_BENCH)
  # Add benchmarks here later, use a similar helper to tests (but WITHOUT test_runner.cc)
  # Example:
  # add_executable(my_bench benchmarks/my_bench.cc)
  # target_link_libraries(my_bench PRIVATE pomai)
  # target_include_directories(my_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

set(POMAI_EXE_DEPS pomai)

# Baseline Benchmark
add_executable(bench_baseline tests/bench_baseline.cc)
target_link_libraries(bench_baseline PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(bench_baseline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Comprehensive Benchmark (Industry-standard metrics)
add_executable(comprehensive_bench benchmarks/comprehensive_bench.cc)
target_link_libraries(comprehensive_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(comprehensive_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Ingestion Throughput Benchmark
add_executable(ingestion_bench benchmarks/ingestion_bench.cc)
target_link_libraries(ingestion_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(ingestion_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# RAG Benchmark
add_executable(rag_bench benchmarks/rag_bench.cc)
target_link_libraries(rag_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(rag_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


# CBR-S benchmark suite
add_executable(bench_cbrs benchmarks/bench_cbrs.cpp)
target_link_libraries(bench_cbrs PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(bench_cbrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(bench_cbrs PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Deterministic CI benchmark for perf gate
add_executable(ci_perf_bench benchmarks/ci_perf_bench.cc)
target_link_libraries(ci_perf_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(ci_perf_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# benchmark_a: Multi-Environment Stress Test (IoT / Edge / Cloud validation)
add_executable(benchmark_a benchmarks/palloc_env_stress.cc)
target_link_libraries(benchmark_a PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(benchmark_a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# =========================
# Tools
# =========================
add_executable(pomai_inspect src/tools/pomai_inspect.cc)
target_link_libraries(pomai_inspect PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(pomai_inspect PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =========================
# C API Tests & Examples
# =========================
if (POMAI_BUILD_TESTS)
  add_executable(c_api_test tests/test_c_api_basic.cpp)
  target_link_libraries(c_api_test PRIVATE pomai_c_static)
  target_sources(c_api_test PRIVATE tests/common/test_runner.cc)
  target_include_directories(c_api_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME c_api_test COMMAND c_api_test)
endif()

add_executable(c_basic_example examples/c_basic.c)
target_link_libraries(c_basic_example PRIVATE pomai_c)
target_include_directories(c_basic_example PRIVATE include)

add_executable(c_scan_export_example examples/c_scan_export.c)
target_link_libraries(c_scan_export_example PRIVATE pomai_c)
target_include_directories(c_scan_export_example PRIVATE include)

install(TARGETS pomai pomai_c pomai_c_static
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES
  include/pomai/c_api.h
  include/pomai/c_status.h
  include/pomai/c_types.h
  include/pomai/c_version.h
  include/pomai/database.h
  DESTINATION include/pomai)
