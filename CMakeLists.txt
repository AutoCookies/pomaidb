cmake_minimum_required(VERSION 3.10)
project(pomai VERSION 1.0.0 LANGUAGES C CXX)

# --- 1. CẤU HÌNH TIÊU CHUẨN C++ ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 2. TÌM KIẾM PHỤ THUỘC (DEPENDENCIES) ---
find_package(Threads REQUIRED)

# --- 3. CẤU HÌNH COMPILER FLAGS (TARGET-BASED) ---
# Tạo một interface library để quản lý compiler flags chung cho toàn dự án
add_library(pomai_compiler_flags INTERFACE)
target_compile_features(pomai_compiler_flags INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(pomai_compiler_flags INTERFACE /W4 /WX)
else()
    target_compile_options(pomai_compiler_flags INTERFACE -Wall -Wextra -Wpedantic -mavx2 -mfma)
    if(CMAKE_BUILD_TYPE MATCHES Release)
        target_compile_options(pomai_compiler_flags INTERFACE -O3 -march=native -ffast-math -funroll-loops)
        # Kích hoạt Interprocedural Optimization (LTO) nếu trình biên dịch hỗ trợ
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

set(POMAI_SANITIZE "" CACHE STRING "Enable sanitizers: ASAN_UBSAN or TSAN")
set_property(CACHE POMAI_SANITIZE PROPERTY STRINGS "" ASAN_UBSAN TSAN)
if(POMAI_SANITIZE STREQUAL "ASAN_UBSAN")
    target_compile_options(pomai_compiler_flags INTERFACE -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    target_link_options(pomai_compiler_flags INTERFACE -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
elseif(POMAI_SANITIZE STREQUAL "TSAN")
    target_compile_options(pomai_compiler_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(pomai_compiler_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
endif()

# --- 4. CORE LIBRARY (pomai_core) ---
# Tiêu chuẩn Big Tech: Liệt kê tường minh các tệp nguồn để tránh lỗi "ma" từ GLOB
set(CORE_SOURCES
    src/core/membrane.cpp
    src/core/pomai_db.cpp
    src/core/shard.cpp
    src/core/wal.cpp
    src/core/wal_uring.cpp
    src/core/seed.cpp
    src/core/spatial_router.cpp
    src/core/orbit_index.cpp
    src/core/memory_manager.cpp
    src/core/whispergrain.cpp
    src/core/crc64.c
)

add_library(pomai_core STATIC ${CORE_SOURCES})

# Cấu hình include theo Target: Không dùng include_directories toàn cục
target_include_directories(pomai_core 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(pomai_core 
    PUBLIC 
        Threads::Threads
        pomai_compiler_flags
)

# --- 5. SERVER DAEMON (pomai-server) ---
set(SERVER_SOURCES
    src/server/server.cpp
    src/server/protocol.cpp
    src/server/pomai_server_main.cpp
    src/server/logger.cpp
    src/server/config.cpp
)

add_executable(pomai-server ${SERVER_SOURCES})
target_include_directories(pomai-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/server)
target_link_libraries(pomai-server PRIVATE pomai_core)

# --- 6. EMBEDDED BENCHMARKS ---
function(add_pomai_benchmark name src)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE pomai_core)
    endif()
endfunction()

add_pomai_benchmark(bench_embedded benchmarks/bench_embedded.cpp)
add_pomai_benchmark(bench_phase3 benchmarks/bench_phase3.cpp)
add_pomai_benchmark(bench_concurrent benchmarks/bench_concurrent.cpp)
add_pomai_benchmark(bench_concurrent_search benchmarks/bench_concurrent_search.cpp)
add_pomai_benchmark(bench_wal bench/bench_wal.cpp)

# --- 7. AUTOMATED TESTS ---
option(BUILD_TESTING "Build the test suite" ON)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()
    
    macro(add_pomai_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE pomai_core)
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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/wal_crash_test.cpp")
    add_executable(wal_crash_test tools/wal_crash_test.cpp)
    target_link_libraries(wal_crash_test PRIVATE pomai_core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/repro_segfault.cpp")
    add_executable(repro_segfault tools/repro_segfault.cpp)
    target_link_libraries(repro_segfault PRIVATE pomai_core)
endif()
