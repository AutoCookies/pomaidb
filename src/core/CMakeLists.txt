# src/core/CMakeLists.txt
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(pomai-core ${CORE_SOURCES})

target_include_directories(pomai-core
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(pomai-core PRIVATE -Wall -Wextra)