cmake_minimum_required(VERSION 3.10)
project(pomai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. TÌM THƯ VIỆN THREADS TRƯỚC (QUAN TRỌNG) ---
find_package(Threads REQUIRED)

# --- 2. CẤU HÌNH COMPILER ---
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
else()
    add_compile_options(-O0 -g)
endif()

include_directories(
    include       
    src 
    src/core 
    src/server 
    .
)

# ==========================================
# 3. CORE LIBRARY (Embedded Mode)
# ==========================================
file(GLOB CORE_SOURCES 
    "src/core/*.cpp"
    "src/core/crc64.c"
)

# Tạo thư viện tĩnh
add_library(pomai_core STATIC ${CORE_SOURCES})
# Link Threads vào Core (Lúc này Threads::Threads đã được tìm thấy ở bước 1)
target_link_libraries(pomai_core Threads::Threads)

# ==========================================
# 4. SERVER DAEMON
# ==========================================
file(GLOB SERVER_SOURCES 
    "src/server/*.cpp"
)

add_executable(pomai-server ${SERVER_SOURCES})
# Link Server với Core
target_link_libraries(pomai-server pomai_core)

# ==========================================
# 5. EMBEDDED BENCHMARK
# ==========================================
# Kiểm tra nếu file tồn tại thì mới build benchmark
if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_embedded.cpp")
    add_executable(bench_embedded benchmarks/bench_embedded.cpp)
    target_link_libraries(bench_embedded pomai_core)
endif()