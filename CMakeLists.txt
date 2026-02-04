cmake_minimum_required(VERSION 3.20)
project(pomai LANGUAGES CXX)

# =========================
# Global build settings
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POMAI_BUILD_TESTS "Build tests" OFF)
option(POMAI_BUILD_BENCH "Build benchmarks" OFF)

# =========================
# Library: pomai
# =========================
add_library(pomai STATIC
  src/api/db.cc
  src/core/engine/engine.cc
  src/core/shard/shard.cc
  src/core/shard/runtime.cc
  src/core/shard/iterator.cc
  src/core/shard/manifest.cc
  src/core/membrane/manager.cc
  src/table/memtable.cc
  src/storage/wal/wal.cc
  src/storage/manifest/manifest.cc
  src/table/segment.cc

  src/core/index/ivf_coarse.cc
  src/core/index/ivf_flat.cc
  src/core/distance.cc

  # Gate #1/#2 utilities (linked into libpomai)
  src/util/posix_file.cc
  src/util/crc32c.cc
)

target_include_directories(pomai
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
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
    # Link against main library
    target_link_libraries(${target} PRIVATE pomai)

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

  add_executable(memtable_test tests/unit/memtable_test.cc)
  pomai_setup_test(memtable_test)
  pomai_add_labeled_test(memtable_test "unit")

  add_executable(mailbox_test tests/unit/mailbox_test.cc)
  pomai_setup_test(mailbox_test)
  pomai_add_labeled_test(mailbox_test "unit")

  add_executable(wal_test tests/unit/wal_test.cc)
  pomai_setup_test(wal_test)
  pomai_add_labeled_test(wal_test "unit")

  add_executable(segment_test tests/unit/segment_test.cc)
  pomai_setup_test(segment_test)
  pomai_add_labeled_test(segment_test "unit")
  
  add_executable(shard_manifest_test tests/unit/shard_manifest_test.cc)
  pomai_setup_test(shard_manifest_test)
  pomai_add_labeled_test(shard_manifest_test "unit")

  add_executable(oracle_test tests/unit/oracle_test.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(oracle_test)
  pomai_add_labeled_test(oracle_test "unit")

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

  add_executable(manifest_test tests/unit/manifest_test.cc)
  pomai_setup_test(manifest_test)
  pomai_add_labeled_test(manifest_test "unit")

  add_executable(membrane_test tests/integ/membrane_test.cc)
  pomai_setup_test(membrane_test)
  pomai_add_labeled_test(membrane_test "integ")

  add_executable(manifest_corruption_test tests/crash/manifest_corruption_test.cc)
  pomai_setup_test(manifest_corruption_test)
  pomai_add_labeled_test(manifest_corruption_test "crash")

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
    set_tests_properties(pomai_crash_replay PROPERTIES LABELS "crash")
  endif()

  # ---- TSAN (Gate #3) ----
  add_executable(db_concurrency_tsan_test tests/tsan/db_concurrency_tsan_test.cc)
  pomai_setup_test(db_concurrency_tsan_test)
  pomai_add_labeled_test(db_concurrency_tsan_test "tsan")

  add_executable(shard_runtime_tsan_test tests/tsan/shard_runtime_tsan_test.cc)
  pomai_setup_test(shard_runtime_tsan_test)
  pomai_add_labeled_test(shard_runtime_tsan_test "tsan")
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

# Baseline Benchmark
add_executable(bench_baseline tests/bench_baseline.cc)
target_link_libraries(bench_baseline PRIVATE pomai)
target_include_directories(bench_baseline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Comprehensive Benchmark (Industry-standard metrics)
add_executable(comprehensive_bench benchmarks/comprehensive_bench.cc)
target_link_libraries(comprehensive_bench PRIVATE pomai)
target_include_directories(comprehensive_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Ingestion Throughput Benchmark
add_executable(ingestion_bench benchmarks/ingestion_bench.cc)
target_link_libraries(ingestion_bench PRIVATE pomai)
target_include_directories(ingestion_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# =========================
# Tools
# =========================
add_executable(pomai_inspect src/tools/pomai_inspect.cc)
target_link_libraries(pomai_inspect PRIVATE pomai)
target_include_directories(pomai_inspect PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
