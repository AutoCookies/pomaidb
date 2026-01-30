cmake_minimum_required(VERSION 3.10)
project(pomai VERSION 1.0.0 LANGUAGES C CXX)

# --- 1. CẤU HÌNH TIÊU CHUẨN C++ ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 2. TÌM KIẾM PHỤ THUỘC (DEPENDENCIES) ---
find_package(Threads REQUIRED)

# --- 3. CẤU HÌNH COMPILER FLAGS (TARGET-BASED) ---
add_library(pomai_compiler_flags INTERFACE)
target_compile_features(pomai_compiler_flags INTERFACE cxx_std_20)

option(POMAI_WERROR "Treat warnings as errors" OFF)
option(POMAI_ASAN "Enable AddressSanitizer" OFF)
option(POMAI_TSAN "Enable ThreadSanitizer" OFF)
option(POMAI_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(MSVC)
    target_compile_options(pomai_compiler_flags INTERFACE /W4)
    if(POMAI_WERROR)
        target_compile_options(pomai_compiler_flags INTERFACE /WX)
    endif()
else()
    target_compile_options(pomai_compiler_flags INTERFACE -Wall -Wextra -Wpedantic -mavx2 -mfma)
    if(POMAI_WERROR)
        target_compile_options(pomai_compiler_flags INTERFACE -Werror)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES Release)
        target_compile_options(pomai_compiler_flags INTERFACE -O3 -march=native -ffast-math -funroll-loops)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT error)
        if(ipo_supported)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
        target_compile_options(pomai_compiler_flags INTERFACE -O2 -g3 -fno-omit-frame-pointer)
    else()
        target_compile_options(pomai_compiler_flags INTERFACE -O1 -g3 -fno-omit-frame-pointer)
    endif()
    target_link_options(pomai_compiler_flags INTERFACE -rdynamic)
endif()

set(pomai_sanitize_flags "")
if(POMAI_ASAN)
    list(APPEND pomai_sanitize_flags address)
endif()
if(POMAI_UBSAN)
    list(APPEND pomai_sanitize_flags undefined)
endif()
if(POMAI_TSAN)
    list(APPEND pomai_sanitize_flags thread)
endif()
if(pomai_sanitize_flags)
    list(JOIN pomai_sanitize_flags "," pomai_sanitize_flags_joined)
    target_compile_options(pomai_compiler_flags INTERFACE -fsanitize=${pomai_sanitize_flags_joined} -fno-omit-frame-pointer)
    target_link_options(pomai_compiler_flags INTERFACE -fsanitize=${pomai_sanitize_flags_joined} -fno-omit-frame-pointer)
endif()

set(POMAI_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- 4. MODULE LIBRARIES ---
set(UTIL_SOURCES
    src/util/memory_manager.cpp
)
add_library(pomai_util STATIC ${UTIL_SOURCES})
target_include_directories(pomai_util PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_util PUBLIC pomai_compiler_flags)

set(STORAGE_SOURCES
    src/storage/wal.cpp
    src/storage/wal_uring.cpp
    src/storage/crc64.c
)
add_library(pomai_storage STATIC ${STORAGE_SOURCES})
target_include_directories(pomai_storage PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_storage PUBLIC pomai_util Threads::Threads pomai_compiler_flags)

set(INDEX_SOURCES
    src/index/orbit_index.cpp
    src/index/whispergrain.cpp
)
add_library(pomai_index STATIC ${INDEX_SOURCES})
target_include_directories(pomai_index PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_index PUBLIC pomai_util pomai_compiler_flags)

set(CORE_SOURCES
    src/core/membrane.cpp
    src/core/seed.cpp
    src/core/shard.cpp
    src/core/spatial_router.cpp
    src/core/wal_seed.cpp
)
add_library(pomai_core STATIC ${CORE_SOURCES})
target_include_directories(pomai_core PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_core PUBLIC pomai_util pomai_storage pomai_index Threads::Threads pomai_compiler_flags)

set(API_SOURCES
    src/api/pomai_db.cpp
)
add_library(pomai_api STATIC ${API_SOURCES})
target_include_directories(pomai_api PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_api PUBLIC pomai_core pomai_compiler_flags)

# --- 5. SERVER DAEMON (pomai-server) ---
set(SERVER_SOURCES
    src/server/server.cpp
    src/server/protocol.cpp
    src/server/pomai_server_main.cpp
    src/server/logger.cpp
    src/server/config.cpp
)

add_executable(pomai-server ${SERVER_SOURCES})
target_link_libraries(pomai-server PRIVATE pomai_api)

# --- 6. EMBEDDED BENCHMARKS ---
function(add_pomai_benchmark name src lib)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE ${lib})
    endif()
endfunction()

add_pomai_benchmark(bench_embedded benchmarks/bench_embedded.cpp pomai_api)
add_pomai_benchmark(bench_concurrent_search benchmarks/bench_concurrent_search.cpp pomai_core)
add_pomai_benchmark(bench_wal benchmarks/bench_wal.cpp pomai_storage)

# --- 7. AUTOMATED TESTS ---
option(BUILD_TESTING "Build the test suite" ON)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()

    macro(add_pomai_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE pomai_api)
        add_test(NAME ${name} COMMAND ${name})
    endmacro()

    add_pomai_test(wal_unit_tests tests/wal_unit_tests.cpp)
    add_pomai_test(wal_durable_tests tests/wal_durable_tests.cpp)
    add_pomai_test(wal_concurrency_tests tests/wal_concurrency_tests.cpp)
    add_pomai_test(wal_integration_writer tests/wal_integration_writer.cpp)
    add_pomai_test(wal_replay_inspect tests/wal_replay_inspect.cpp)
    add_pomai_test(pomai_db_writer tests/pomai_db_writer.cpp)
    add_pomai_test(centroids_persistence_tests tests/centroids_persistence_tests.cpp)
    add_pomai_test(memory_safety_tests tests/memory_safety_tests.cpp)
    add_pomai_test(quantization_fixed_bounds_tests tests/quantization_fixed_bounds_tests.cpp)
    add_pomai_test(rerank_exact_tests tests/rerank_exact_tests.cpp)
    add_pomai_test(shard_concurrency_tests tests/shard_concurrency_tests.cpp)
    add_pomai_test(pomai_shutdown_concurrency_tests tests/pomai_shutdown_concurrency_tests.cpp)
    add_pomai_test(wal_append_regression_tests tests/wal_append_regression_tests.cpp)
    add_pomai_test(tsan_smoke_tests tests/tsan_smoke_tests.cpp)
endif()

# --- 8. TOOLS ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/repro_segfault.cpp")
    add_executable(repro_segfault tools/repro_segfault.cpp)
    target_link_libraries(repro_segfault PRIVATE pomai_api)
endif()
