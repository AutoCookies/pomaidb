cmake_minimum_required(VERSION 3.16)
project(pomai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(pomai-server
  src/server/pomai_server_main.cpp
  src/server/config.cpp
  src/server/logger.cpp
  src/server/protocol.cpp
  src/server/server.cpp

  src/core/wal.cpp
  src/core/seed.cpp
  src/core/shard.cpp
  src/core/membrane.cpp
  src/core/pomai_db.cpp
)

target_include_directories(pomai-server PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pomai-server PRIVATE Threads::Threads)
target_compile_options(pomai-server PRIVATE -Wall -Wextra -Wpedantic)
