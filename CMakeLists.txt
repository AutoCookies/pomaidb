cmake_minimum_required(VERSION 3.20)

# ---- Project ----
project(pomai
  VERSION 0.1.0
  DESCRIPTION "Pomai: embedded, multithreaded vector database (share-nothing shards)"
  LANGUAGES CXX
)

# ---- Options (bigtech style toggles) ----
option(POMAI_BUILD_EXAMPLES "Build examples" OFF)
option(POMAI_BUILD_TESTS "Build tests (CTest)" OFF)
option(POMAI_BUILD_BENCHMARKS "Build benchmarks" ON)

option(POMAI_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(POMAI_ENABLE_LTO "Enable Link-Time Optimization (Release/RelWithDebInfo)" ON)

option(POMAI_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(POMAI_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(POMAI_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

option(POMAI_ENABLE_CLANG_TIDY "Enable clang-tidy if available" OFF)

# ---- Global C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Output directories ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Default build type ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Helpers ----
include(CheckIPOSupported)

function(pomai_enable_warnings TARGET_NAME)
  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
    if(POMAI_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET_NAME} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${TARGET_NAME} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wshadow
      -Wconversion
      -Wsign-conversion
      -Wold-style-cast
      -Woverloaded-virtual
      -Wnon-virtual-dtor
      -Wformat=2
      -Wnull-dereference
      -Wduplicated-cond
      -Wduplicated-branches
      -Wlogical-op
      -Wuseless-cast
    )
    if(POMAI_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET_NAME} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(pomai_enable_sanitizers TARGET_NAME)
  if(MSVC)
    message(STATUS "Sanitizers are not configured for MSVC in this template.")
    return()
  endif()

  # TSAN must not be combined with ASAN.
  if(POMAI_ENABLE_TSAN)
    target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(${TARGET_NAME} PRIVATE -fsanitize=thread)
  else()
    if(POMAI_ENABLE_ASAN)
      target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address)
    endif()
    if(POMAI_ENABLE_UBSAN)
      target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
      target_link_options(${TARGET_NAME} PRIVATE -fsanitize=undefined)
    endif()
  endif()
endfunction()

function(pomai_enable_lto TARGET_NAME)
  if(NOT POMAI_ENABLE_LTO)
    return()
  endif()
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO/LTO not supported: ${ipo_error}")
  endif()
endfunction()

# ---- clang-tidy (optional) ----
if(POMAI_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  else()
    message(STATUS "clang-tidy requested but not found")
  endif()
endif()

# ---- Threads ----
find_package(Threads REQUIRED)

# =============================================================================
# Library: pomai (embedded core)
# =============================================================================

add_library(pomai STATIC
  src/pomai.cpp
  src/router.cpp
  src/shard.cpp
  src/wal.cpp
  src/util/crc32c.cpp
)

target_include_directories(pomai
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pomai
  PUBLIC
    Threads::Threads
)

# CPU opts (keep it sane; user can opt into -march=native outside)
if(NOT MSVC)
  target_compile_options(pomai PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>
  )
endif()

pomai_enable_warnings(pomai)
pomai_enable_sanitizers(pomai)
pomai_enable_lto(pomai)

# =============================================================================
# Benchmarks
# =============================================================================

if(POMAI_BUILD_BENCHMARKS)
  add_executable(bench_embedded_upsert benchmarks/bench_embedded_upsert.cpp)
  target_link_libraries(bench_embedded_upsert PRIVATE pomai)
  pomai_enable_warnings(bench_embedded_upsert)
  pomai_enable_sanitizers(bench_embedded_upsert)
  pomai_enable_lto(bench_embedded_upsert)
endif()

# =============================================================================
# Examples
# =============================================================================

if(POMAI_BUILD_EXAMPLES)
  add_executable(basic examples/basic.cpp)
  target_link_libraries(basic PRIVATE pomai)
  pomai_enable_warnings(basic)
  pomai_enable_sanitizers(basic)
endif()

# =============================================================================
# Tests (placeholder)
# =============================================================================

if(POMAI_BUILD_TESTS)
  enable_testing()
  message(STATUS "POMAI_BUILD_TESTS=ON but no tests are defined yet.")
endif()

# =============================================================================
# Install (optional)
# =============================================================================

include(GNUInstallDirs)

install(TARGETS pomai
  EXPORT pomaiTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT pomaiTargets
  FILE pomaiTargets.cmake
  NAMESPACE pomai::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pomai
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "Pomai build configuration:")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build benchmarks:     ${POMAI_BUILD_BENCHMARKS}")
message(STATUS "  Build examples:       ${POMAI_BUILD_EXAMPLES}")
message(STATUS "  Build tests:          ${POMAI_BUILD_TESTS}")
message(STATUS "  Warnings as errors:   ${POMAI_WARNINGS_AS_ERRORS}")
message(STATUS "  LTO enabled:          ${POMAI_ENABLE_LTO}")
message(STATUS "  ASAN:                 ${POMAI_ENABLE_ASAN}")
message(STATUS "  UBSAN:                ${POMAI_ENABLE_UBSAN}")
message(STATUS "  TSAN:                 ${POMAI_ENABLE_TSAN}")
message(STATUS "  clang-tidy:           ${POMAI_ENABLE_CLANG_TIDY}")
message(STATUS "")
