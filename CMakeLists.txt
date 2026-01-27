cmake_minimum_required(VERSION 3.10)
project(pomai CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 1. PERFORMANCE FLAGS (CRITICAL FOR AVX2) ---
# Tự động phát hiện kiến trúc CPU (Haswell, Skylake, etc.) để bật AVX2/AVX-512
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -march=native: Bật tất cả tập lệnh mà CPU hiện tại hỗ trợ (AVX2, FMA, SSE4.2...)
    # -O3: Tối ưu hóa tối đa
    add_compile_options(-march=native -O3 -Wall -Wextra -pthread)
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX2)
endif()

# --- 2. INCLUDE PATHS ---
# Vì bạn đã chuyển file header vào cùng source hoặc ra ngoài include/pomai
# Ta thêm trực tiếp các thư mục src vào đường dẫn tìm header.
include_directories(
    src
    src/core
    include # Fallback nếu vẫn còn file trong này
)

# --- 3. SOURCE FILES ---
# Quét toàn bộ file source. 
# Lưu ý: Trong Production lớn, ta nên liệt kê từng file. 
# Nhưng với Pomai hiện tại, GLOB_RECURSE giúp bạn refactor file thoải mái không cần sửa cmake.
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cc" 
    "src/*.c"
)

# --- 4. EXECUTABLE ---
add_executable(pomai-server ${SOURCES})

# Link thư viện (nếu cần, hiện tại pthread đã được add qua compile options)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(pomai-server atomic) 
endif()

message(STATUS "Pomai Build Configured.")
message(STATUS "  - AVX2/FMA: Auto-enabled via -march=native")
message(STATUS "  - Sources found: ${SOURCES}")