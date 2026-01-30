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
option(POMAI_BUILD_TESTS "Build the test suite" ON)
option(POMAI_ASAN "Enable AddressSanitizer" OFF)
option(POMAI_TSAN "Enable ThreadSanitizer" OFF)
option(POMAI_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
set(POMAI_SANITIZE "" CACHE STRING "Sanitizer to enable (ASAN, UBSAN, TSAN)")
set_property(CACHE POMAI_SANITIZE PROPERTY STRINGS "" ASAN UBSAN TSAN)
if(POMAI_SANITIZE STREQUAL "ASAN")
    set(POMAI_ASAN ON)
elseif(POMAI_SANITIZE STREQUAL "UBSAN")
    set(POMAI_UBSAN ON)
elseif(POMAI_SANITIZE STREQUAL "TSAN")
    set(POMAI_TSAN ON)
endif()

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
set(CONCURRENCY_SOURCES
    src/concurrency/memory_manager.cpp
    src/util/logger.cpp
)
add_library(pomai_concurrency STATIC ${CONCURRENCY_SOURCES})
target_include_directories(pomai_concurrency PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_concurrency PUBLIC pomai_compiler_flags)

set(STORAGE_SOURCES
    src/storage/file_util.cpp
    src/storage/manifest.cpp
    src/storage/snapshot.cpp
    src/storage/verify.cpp
    src/storage/wal.cpp
    src/storage/wal_uring.cpp
    src/storage/crc64.c
)
add_library(pomai_storage STATIC ${STORAGE_SOURCES})
target_include_directories(pomai_storage PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_storage PUBLIC pomai_concurrency Threads::Threads pomai_compiler_flags)

set(INDEX_SOURCES
    src/index/orbit_index.cpp
    src/index/whispergrain.cpp
)
add_library(pomai_index STATIC ${INDEX_SOURCES})
target_include_directories(pomai_index PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_index PUBLIC pomai_concurrency pomai_compiler_flags)

set(CORE_SOURCES
    src/core/membrane.cpp
    src/core/seed.cpp
    src/core/shard.cpp
    src/core/spatial_router.cpp
    src/core/wal_seed.cpp
)
add_library(pomai_core STATIC ${CORE_SOURCES})
target_include_directories(pomai_core PUBLIC $<BUILD_INTERFACE:${POMAI_PUBLIC_INCLUDE_DIR}>)
target_link_libraries(pomai_core PUBLIC pomai_concurrency pomai_storage pomai_index Threads::Threads pomai_compiler_flags)

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
if(POMAI_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()
    add_subdirectory(tests)
endif()

# --- 8. TOOLS ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/repro_segfault.cpp")
    add_executable(repro_segfault src/tools/repro_segfault.cpp)
    target_link_libraries(repro_segfault PRIVATE pomai_api)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/pomai_verify.cpp")
    add_executable(pomai_verify src/tools/pomai_verify.cpp)
    target_link_libraries(pomai_verify PRIVATE pomai_api)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/pomai_dump.cpp")
    add_executable(pomai_dump src/tools/pomai_dump.cpp)
    target_link_libraries(pomai_dump PRIVATE pomai_api)
endif()
