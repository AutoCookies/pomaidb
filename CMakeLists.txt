cmake_minimum_required(VERSION 3.20)
project(pomai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POMAI_BUILD_TESTS "Build tests" OFF)
option(POMAI_BUILD_BENCH "Build benchmarks" OFF)

add_library(pomai STATIC
  src/api/db.cc
  src/core/engine/engine.cc
  src/core/shard/shard.cc
  src/core/shard/runtime.cc
  src/storage/wal/wal.cc
  src/storage/manifest/manifest.cc
  src/table/memtable.cc
  src/core/membrane/manager.cc

  # Gate #1/#2 utilities (must be linked into libpomai)
  src/util/posix_file.cc
  src/util/crc32c.cc
)


target_include_directories(pomai
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =========================
# Tests
# =========================
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
    target_link_libraries(${target} PRIVATE pomai)

    # One main per test binary.
    target_sources(${target} PRIVATE tests/common/test_runner.cc)

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

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

  # ---- Integration (embedded DB API) ----
  add_executable(db_basic_test tests/integ/db_basic_test.cc)
  pomai_setup_test(db_basic_test)
  pomai_add_labeled_test(db_basic_test "integ")

  add_executable(db_persistence_test tests/integ/db_persistence_test.cc)
  pomai_setup_test(db_persistence_test)
  pomai_add_labeled_test(db_persistence_test "integ")

  # ---- Crash (Gate #2) ----
  if (POMAI_ENABLE_CRASH_TESTS)
    add_executable(pomai_crash_test tests/crash/crash_replay_test.cc)
    pomai_setup_test(pomai_crash_test)
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

if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
