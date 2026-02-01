cmake_minimum_required(VERSION 3.20)
project(pomai_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kéo core từ root (path ..)
add_subdirectory(.. pomai_core_build)

add_executable(bench_pomai bench_pomai.cpp)
target_link_libraries(bench_pomai PRIVATE pomai)

# Output gọn
set_target_properties(bench_pomai PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
