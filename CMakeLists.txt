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
if (POMAI_BUILD_TESTS)
  enable_testing()

  # ---- Crash / WAL replay test (Gate #2) ----
  add_executable(pomai_crash_test
    tests/crash/crash_replay_test.cc
  )
  target_link_libraries(pomai_crash_test PRIVATE pomai)
  target_compile_options(pomai_crash_test PRIVATE -Wall -Wextra -Wpedantic)

  add_test(
    NAME pomai_crash_replay
    COMMAND pomai_crash_test
  )
  set_tests_properties(pomai_crash_replay PROPERTIES LABELS "crash")

  # ---- Concurrency / TSAN-friendly tests (Gate #3) ----
  # (Chỉ bật khi mày có file test tương ứng)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/tsan/concurrency_test.cc")
    add_executable(pomai_concurrency_test
      tests/tsan/concurrency_test.cc
    )
    target_link_libraries(pomai_concurrency_test PRIVATE pomai)
    target_compile_options(pomai_concurrency_test PRIVATE -Wall -Wextra -Wpedantic)

    add_test(
      NAME pomai_concurrency_test
      COMMAND pomai_concurrency_test
    )
    set_tests_properties(pomai_concurrency_test PROPERTIES LABELS "tsan")
  endif()
endif()


if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()
