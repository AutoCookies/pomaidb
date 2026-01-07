# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

enable_testing()

# Example: integration test executable
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cc"
)

# Build each test file as an executable (simple pattern)
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  # Link against project libraries as needed, for example:
  # target_link_libraries(${test_name} PRIVATE pomai-ai pomai-core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()