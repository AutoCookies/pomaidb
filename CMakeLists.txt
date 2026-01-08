# Top-level CMake for the repository.
cmake_minimum_required(VERSION 3.16)
project(pomai_cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optionally allow building tests and apps
option(BUILD_TESTS "Build tests" ON)
option(BUILD_APPS "Build apps/binaries" ON)

# Add top-level subdirectories if they exist. This keeps the top-level
# CMakeLists small while allowing per-module CMake files.
if(EXISTS "${CMAKE_SOURCE_DIR}/src/ai/CMakeLists.txt")
  add_subdirectory(src/ai)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/core/CMakeLists.txt")
  add_subdirectory(src/core)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/memory/CMakeLists.txt")
  add_subdirectory(src/memory)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/facade/CMakeLists.txt")
  add_subdirectory(src/facade)
endif()

if(BUILD_APPS AND EXISTS "${CMAKE_SOURCE_DIR}/apps/CMakeLists.txt")
  add_subdirectory(apps)
endif()

if(BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

# Install rules, packaging, etc. could go here.