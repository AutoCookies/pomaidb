cmake_minimum_required(VERSION 3.18)
project(palloc-test C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
  if ("${CMAKE_BINARY_DIR}" MATCHES ".*(D|d)ebug$")
    message(STATUS "No build type selected, default to *** Debug ***")
    set(CMAKE_BUILD_TYPE "Debug")
  else()
    message(STATUS "No build type selected, default to *** Release ***")
    set(CMAKE_BUILD_TYPE "Release")
  endif()
endif()

# Import palloc (if installed)
find_package(palloc CONFIG REQUIRED)
message(STATUS "Found palloc installed at: ${PALLOC_LIBRARY_DIR} (${PALLOC_VERSION_DIR})")


# link with a dynamic shared library
# use `LD_PRELOAD` to actually override malloc/free at runtime with palloc
add_executable(dynamic-override  main-override.c)
target_link_libraries(dynamic-override PUBLIC palloc)

add_executable(dynamic-override-cxx  main-override.cpp)
target_link_libraries(dynamic-override-cxx PUBLIC palloc)


# overriding with a static object file works reliable as the symbols in the
# object file have priority over those in library files
add_executable(static-override-obj main-override.c ${PALLOC_OBJECT_DIR}/palloc${CMAKE_C_OUTPUT_EXTENSION})
target_include_directories(static-override-obj PUBLIC ${PALLOC_INCLUDE_DIR})
target_link_libraries(static-override-obj PUBLIC palloc-static)


# overriding with a static library works too if using the `palloc-override.h`
# header to redefine malloc/free. (the library already overrides new/delete)
add_executable(static-override-static main-override-static.c)
target_link_libraries(static-override-static PUBLIC palloc-static)


# overriding with a static library: this may not work if the library is linked too late
# on the command line after the C runtime library; but we cannot control that well in CMake
add_executable(static-override main-override.c)
target_link_libraries(static-override PUBLIC palloc-static)

add_executable(static-override-cxx  main-override.cpp)
target_link_libraries(static-override-cxx PUBLIC palloc-static)


## test memory errors
add_executable(test-wrong  test-wrong.c)
target_link_libraries(test-wrong PUBLIC palloc)
