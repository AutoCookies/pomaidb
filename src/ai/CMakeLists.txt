# src/ai/CMakeLists.txt
# Build the ai library (headers under src/ai/include or include/ai)
cmake_minimum_required(VERSION 3.16)

# Collect sources (adjust patterns to match your tree)
file(GLOB_RECURSE AI_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Prefer public headers under include/ai (project-wide include dir).
# If your headers are in src/ai/include, include that as PUBLIC as well.
add_library(pomai-ai ${AI_SOURCES})

target_include_directories(pomai-ai
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Example: link to other internal libs if needed:
# target_link_libraries(pomai-ai PUBLIC pomai-core)

# Enable warnings (optional)
target_compile_options(pomai-ai PRIVATE -Wall -Wextra -Wpedantic)