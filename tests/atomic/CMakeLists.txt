# tests/atomic/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(pomai_atomic_tests)

add_executable(atomic_mmap_test atomic_mmap_test.cc)

target_include_directories(atomic_mmap_test PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# pthread is required for threads on some systems
find_package(Threads REQUIRED)
target_link_libraries(atomic_mmap_test PRIVATE Threads::Threads)