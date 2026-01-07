cmake_minimum_required(VERSION 3.15)
project(cache_system LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build tests" ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(cache INTERFACE)
target_include_directories(cache INTERFACE ${PROJECT_SOURCE_DIR}/include)

add_executable(example examples/init_example.cpp)
target_link_libraries(example PRIVATE cache)

if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(test_init tests/test_init.cpp)
  target_link_libraries(test_init PRIVATE cache GTest::gtest_main pthread)
  add_test(NAME init_tests COMMAND test_init)
endif()