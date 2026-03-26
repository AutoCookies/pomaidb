cmake_minimum_required(VERSION 3.20)
project(pomai VERSION 0.1.0 LANGUAGES C CXX)

# =========================
# Global build settings
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(POMAI_BUILD_TESTS "Build tests" ON)
option(POMAI_BUILD_BENCH "Build benchmarks" OFF)
option(POMAI_REGISTER_BENCH_CTEST "Register benchmark executables as CTest (ctest -L bench)" ON)

# Prefer integer (SQ8/FP16) distance paths where data is quantized; reduces float use on embedded.
option(POMAI_PREFER_INTEGER_MATH "Prefer integer/SQ8/FP16 paths for distance (embedded)" ON)

# Edge-oriented build profile for constrained devices.
# When enabled, we bias towards smaller binaries and lower memory/debug overhead.
option(POMAI_EDGE_BUILD "Optimize PomaiDB build for edge devices (size/footprint)" OFF)

# TFLite build options: minimize dependencies
add_compile_definitions(TFLITE_WITH_RUY=0)
add_compile_definitions(TFLITE_WITH_XNNPACK=0)
add_compile_definitions(TFLITE_WITHOUT_XNNPACK=1)
add_compile_definitions(TFLITE_USE_APPLE_ACCELERATE=0)
add_compile_definitions(TFLITE_WITH_DELEGATES=0)

# Strict build mode: treat project warnings as errors (useful for CI/hardening).
# We keep a couple of warning categories non-fatal to tolerate intentional diagnostics
# from some vendored headers included by the core.
option(POMAI_STRICT "Treat warnings as errors for PomaiDB code" OFF)

# Ensure POMAI_PREFER_INTEGER_MATH is defined for source code (0 or 1)
add_compile_definitions(POMAI_PREFER_INTEGER_MATH=$<BOOL:${POMAI_PREFER_INTEGER_MATH}>)

if (POMAI_EDGE_BUILD)
  add_compile_definitions(POMAI_EDGE_BUILD=1)
  # Favor size over speed and strip most debug info in edge builds.
  add_compile_options(-Os -g0)
endif()


# =========================
# Native HNSW (Replaces FAISS HNSW)
# =========================
message(STATUS "[pomai] Native HNSW: enabled")

# Khronos Vulkan-Headers + Vulkan-Hpp (vendored under third_party/vulkan/include).
# Override with: cmake -DPOMAI_VULKAN_HEADERS_DIR=/path/to/include
set(POMAI_VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include" CACHE PATH
    "Directory containing vulkan/vulkan.h (parent of the vulkan/ header folder)")
if(NOT EXISTS "${POMAI_VULKAN_HEADERS_DIR}/vulkan/vulkan.h")
  message(FATAL_ERROR
    "Vulkan headers not found at:\n  ${POMAI_VULKAN_HEADERS_DIR}/vulkan/vulkan.h\n"
    "Place the Khronos bundle under third_party/vulkan/include/ or set POMAI_VULKAN_HEADERS_DIR.")
endif()
message(STATUS "[pomai] Vulkan headers: ${POMAI_VULKAN_HEADERS_DIR}")

# Vendored palloc (AutoCookies fork): use explicit pa_* API only — do not export
# malloc/free aliases from libpalloc (avoids clashes with libc and other deps).
set(PA_OVERRIDE OFF CACHE BOOL "palloc: interpose libc malloc/free" FORCE)
set(PA_BUILD_SHARED OFF CACHE BOOL "palloc: build shared library" FORCE)
set(PA_BUILD_STATIC ON CACHE BOOL "palloc: build static library" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "palloc: build palloc unit tests" FORCE)
set(PA_BUILD_OBJECT OFF CACHE BOOL "palloc: build palloc-obj target" FORCE)
add_subdirectory(third_party/palloc)

find_package(Threads REQUIRED)

# =========================
# Library: pomai
# =========================
add_library(pomai STATIC
  src/db.cc
  src/database.cc
  src/core/rag/agent_memory.cc
  src/palloc_page_pool.cc
  src/core/vector_engine/vector_engine.cc
  src/core/shard/runtime.cc
  src/core/query/query_planner.cc
  src/core/query/query_orchestrator.cc
  src/core/query/heuristic_engine.cc
  src/core/text/text_membrane.cc
  src/core/timeseries/timeseries_engine.cc
  src/core/keyvalue/keyvalue_engine.cc
  src/core/sketch/sketch_engine.cc
  src/core/blob/blob_engine.cc
  src/core/simd/simd_dispatch.cc
  src/core/spatial/spatial_engine.cc
  src/core/mesh/mesh_engine.cc
  src/core/mesh/mesh_lod_manager.cc
  src/core/mesh/simplifier.cc
  src/core/linker/object_linker.cc
  src/core/connectivity/edge_gateway.cc
  src/core/sparse/sparse_engine.cc
  src/core/bitset/bitset_engine.cc
  src/core/audio/audio_engine.cc
  src/core/bloom/bloom_engine.cc
  src/core/document/document_engine.cc
  src/core/ai/no_train_dispatch.cc
  src/core/lifecycle/semantic_lifecycle.cc
  src/core/security/aes_gcm.cc
  src/core/shard/layer_lookup.cc
  src/core/shard/iterator.cc
  src/core/shard/manifest.cc
  src/core/membrane/manager.cc
  src/core/membrane/membrane_record_iterator.cc
  src/table/memtable.cc
  src/storage/wal/wal.cc
  src/core/kernel/pods/vector_pod.cc
  src/core/kernel/pods/graph_pod.cc
  src/core/kernel/pods/query_pod.cc
  src/storage/manifest/manifest.cc
  src/storage/storage_engine.cc
  src/table/segment.cc
  src/core/storage/compaction_manager.cc
  src/core/storage/sync_provider.cc
  src/core/concurrency/scheduler.cc
  src/compute/vulkan/loader/volk.c
  src/compute/vulkan/vulkan_init.cc
  src/compute/vulkan/vulkan_device_context.cc
  src/compute/vulkan/vulkan_memory_bridge.cc
  src/compute/vulkan/vulkan_staging_pool.cc

  src/core/quantization/scalar_quantizer.cc
  src/core/quantization/half_float_quantizer.cc
  src/core/quantization/bit_quantizer.cc
  src/core/index/ivf_coarse.cc
  src/core/index/ivf_flat.cc
  src/core/distance.cc
  src/core/query/lexical_index.cc

  src/core/routing/routing_table.cc
  src/core/routing/kmeans_lite.cc
  src/core/routing/routing_persist.cc
  src/core/rag/rag_engine.cc
  src/core/rag/chunking/recursive_splitter.cc
  src/core/rag/chunking/sliding_window.cc
  src/core/rag/mock_embedding_provider.cc
  src/core/rag/rag_pipeline.cc

  # Phase 2: Product Quantizer
  src/core/quantization/pomai_pq.cc

  # Phase 3 (Milvus-Lite inspired): BitsetMask pre-computed filter bitset
  src/core/bitset_mask.cc

  src/core/index/hnsw_index.cc
  third_party/pomaidb_hnsw/hnsw.cc

  # Gate #1/#2 utilities (linked into libpomai)
  src/util/posix_file.cc
  src/util/posix_env.cc
  src/util/memory_env.cc
  src/util/env.cc
  src/util/crc32c.cc
  src/util/logging.cc

  # SLM and Tensor Core (Phase 12 Integration)
  src/ai/ai_engine.cpp
  src/ai/cheesebrain_core/ggml.c
  src/ai/cheesebrain_core/ggml-alloc.c
  src/ai/cheesebrain_core/ggml-backend.cpp
  src/ai/cheesebrain_core/ggml-quants.c
  src/ai/cheesebrain_core/gguf.cpp
  src/ai/cheesebrain_core/cheese-model-loader.cpp
  src/ai/cheesebrain_core/cheese-context.cpp
  src/ai/tensor_core/tensorflow/compiler/mlir/lite/core/model_builder_base.cc
  src/ai/tensor_core/tensorflow/lite/core/interpreter.cc
  src/ai/tensor_core/tensorflow/lite/core/interpreter_builder.cc
  src/ai/tensor_core/tensorflow/lite/core/subgraph.cc
  src/ai/tensor_core/tensorflow/compiler/mlir/lite/allocation.cc
  src/ai/tensor_core/tensorflow/lite/core/model_building.cc
  src/ai/tensor_core/tensorflow/lite/kernels/add.cc
  src/ai/tensor_core/tensorflow/lite/kernels/mul.cc
  src/ai/tensor_core/tensorflow/lite/kernels/sub.cc
  src/ai/tensor_core/tensorflow/lite/kernels/reshape.cc
  src/ai/tensor_core/tensorflow/lite/core/kernels/register.cc
  src/ai/analytical_engine.cpp
)

target_include_directories(pomai
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${POMAI_VULKAN_HEADERS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compute/vulkan/loader
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai/cheesebrain_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ai/tensor_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if (WIN32)
  target_compile_definitions(pomai PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# Vendored code: keep strict warnings for PomaiDB, but avoid breaking builds on
# third_party sources when POMAI_STRICT is enabled.
if (POMAI_STRICT AND NOT MSVC)
  set_source_files_properties(third_party/pomaidb_hnsw/hnsw.cc PROPERTIES COMPILE_OPTIONS
    "-Wno-error;-Wno-shadow;-Wno-unused-parameter;-Wno-unused-but-set-variable")
endif()

# OpenMP for parallel builds if needed
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  target_link_libraries(pomai PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(OpenSSL REQUIRED)
# PUBLIC: static lib pomai leaves unresolved pa_* until consumers link palloc-static too.
target_link_libraries(pomai PUBLIC palloc-static Threads::Threads)
target_link_libraries(pomai PRIVATE OpenSSL::Crypto)
if (UNIX AND NOT APPLE)
  target_link_libraries(pomai PRIVATE ${CMAKE_DL_LIBS})
endif()

if (MSVC)
  target_compile_options(pomai PRIVATE /W4 /permissive-)
else()
  # Strict warnings for pomaidb code; then suppress vendored third_party (simd) warnings.
  # We do not use / -Werror globally because some third_party headers intentionally emit #warning.
  target_compile_options(pomai PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
    -Wno-cpp -Wno-unknown-pragmas -Wno-conversion -Wno-float-conversion -Wno-unused-function
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai PRIVATE /WX)
  else()
    target_compile_options(pomai PRIVATE
      -Werror
      # Some vendored headers use extensions that trigger pedantic/cpp warnings.
      -Wno-error=pedantic
      -Wno-error=cpp
      # third_party/simd uses #warning; pre-C++23 this triggers -Wc++23-extensions.
      # We disable that warning category entirely to keep strict builds usable.
      -Wno-c++23-extensions
    )
  endif()
endif()


# =========================
# Library: pomai_c (C ABI)
# =========================
set(POMAI_C_API_SOURCES
  src/capi/capi_status.cc
  src/capi/capi_db.cc
  src/capi/capi_scan.cc
  src/capi/capi_membrane_scan.cc
  src/capi/capi_membrane_capabilities.cc
  src/capi/capi_agent_memory.cc
)

add_library(pomai_c SHARED ${POMAI_C_API_SOURCES})
add_library(pomai_c_static STATIC ${POMAI_C_API_SOURCES})

set_target_properties(pomai_c_static PROPERTIES OUTPUT_NAME pomai_c)

target_link_libraries(pomai_c PRIVATE pomai)
# Static C API archive must re-export pomai usage requirements (e.g. palloc headers)
# so targets that include C++ headers such as pomai/database.h can compile.
target_link_libraries(pomai_c_static PUBLIC pomai)

# Export C API symbols when building shared library on Windows.
target_compile_definitions(pomai_c PRIVATE POMAI_C_BUILD_DLL)

target_include_directories(pomai_c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(pomai_c_static
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
  target_compile_options(pomai_c PRIVATE /W4 /permissive-)
  target_compile_options(pomai_c_static PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomai_c PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
  target_compile_options(pomai_c_static PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wunused-parameter -Wunused-variable
  )
endif()

if (POMAI_STRICT)
  if (MSVC)
    target_compile_options(pomai_c PRIVATE /WX)
    target_compile_options(pomai_c_static PRIVATE /WX)
  else()
    target_compile_options(pomai_c PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
    target_compile_options(pomai_c_static PRIVATE -Werror -Wno-error=pedantic -Wno-error=cpp -Wno-c++23-extensions)
  endif()
endif()

# =========================
# Tests
# =========================
if (POMAI_BUILD_TESTS)
  enable_testing()

  # Detect TSAN builds (best-effort).
  set(POMAI_IS_TSAN OFF)
  if (CMAKE_CXX_FLAGS MATCHES "fsanitize=thread" OR CMAKE_EXE_LINKER_FLAGS MATCHES "fsanitize=thread")
    set(POMAI_IS_TSAN ON)
  endif()

  option(POMAI_ENABLE_CRASH_TESTS "Enable crash-style tests (may be incompatible with TSAN)" ON)
  if (POMAI_IS_TSAN)
    set(POMAI_ENABLE_CRASH_TESTS OFF)
  endif()

  function(pomai_setup_test target)
    # Link against main C++ library and C API shim
    target_link_libraries(${target} PRIVATE pomai pomai_c)

    # One main per test binary (your harness)
    target_sources(${target} PRIVATE tests/common/test_runner.cc)

    # Warnings
    if (MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    # Allow tests to include both public and internal headers.
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}              # "tests/common/..."
      ${CMAKE_CURRENT_SOURCE_DIR}/include      # public API
      ${CMAKE_CURRENT_SOURCE_DIR}/src          # internal headers: util/, core/, table/, storage/
    )

    # Make TSAN builds more stable on some distros (PIE can trigger weird mmap layouts).
    if (POMAI_IS_TSAN)
      target_compile_options(${target} PRIVATE -fno-pie)
      target_link_options(${target} PRIVATE -no-pie)
    endif()
  endfunction()

  function(pomai_add_labeled_test target label)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${label}")
  endfunction()

  # ---- Unit ----
  add_executable(crc32c_test tests/unit/crc32c_test.cc)
  pomai_setup_test(crc32c_test)
  pomai_add_labeled_test(crc32c_test "unit")

  add_executable(hnsw_pomaidb_test tests/unit/hnsw_pomaidb_test.cc)
  pomai_setup_test(hnsw_pomaidb_test)
  pomai_add_labeled_test(hnsw_pomaidb_test "unit")

  add_executable(memtable_test tests/unit/memtable_test.cc)
  pomai_setup_test(memtable_test)
  pomai_add_labeled_test(memtable_test "unit")

  add_executable(palloc_page_pool_test tests/unit/palloc_page_pool_test.cc)
  pomai_setup_test(palloc_page_pool_test)
  pomai_add_labeled_test(palloc_page_pool_test "unit")

  add_executable(env_test tests/unit/env_test.cc)
  pomai_setup_test(env_test)
  pomai_add_labeled_test(env_test "unit")

  add_executable(agent_memory_core_test tests/unit/agent_memory_core_test.cc)
  pomai_setup_test(agent_memory_core_test)
  # Temporarily exclude from default CTest runs until AgentMemory semantics are finalized.
  # pomai_add_labeled_test(agent_memory_core_test "unit")

  add_executable(agent_memory_concurrency_test tests/unit/agent_memory_concurrency_test.cc)
  pomai_setup_test(agent_memory_concurrency_test)
  pomai_add_labeled_test(agent_memory_concurrency_test "unit")

  add_executable(mailbox_test tests/unit/mailbox_test.cc)
  pomai_setup_test(mailbox_test)
  pomai_add_labeled_test(mailbox_test "unit")

  add_executable(logging_test tests/unit/logging_test.cc)
  pomai_setup_test(logging_test)
  pomai_add_labeled_test(logging_test "unit")

  add_executable(wal_test tests/unit/wal_test.cc)
  pomai_setup_test(wal_test)
  pomai_add_labeled_test(wal_test "unit")

  add_executable(segment_test tests/unit/segment_test.cc)
  pomai_setup_test(segment_test)
  pomai_add_labeled_test(segment_test "unit")

  add_executable(fp16_test tests/unit/fp16_test.cc)
  pomai_setup_test(fp16_test)
  pomai_add_labeled_test(fp16_test "unit")
  
  add_executable(shard_manifest_test tests/unit/shard_manifest_test.cc)
  pomai_setup_test(shard_manifest_test)
  pomai_add_labeled_test(shard_manifest_test "unit")

  add_executable(oracle_test tests/unit/oracle_test.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(oracle_test)
  pomai_add_labeled_test(oracle_test "unit")

  add_executable(routing_table_test tests/unit/routing_table_test.cc)
  pomai_setup_test(routing_table_test)
  pomai_add_labeled_test(routing_table_test "unit")

  add_executable(streaming_index_test tests/unit/streaming_index_test.cc)
  pomai_setup_test(streaming_index_test)
  pomai_add_labeled_test(streaming_index_test "unit")

  add_executable(edge_platform_test tests/unit/edge_platform_test.cc)
  pomai_setup_test(edge_platform_test)
  pomai_add_labeled_test(edge_platform_test "unit")

  add_executable(graph_rag_test tests/unit/graph_rag_test.cc)
  pomai_setup_test(graph_rag_test)
  pomai_add_labeled_test(graph_rag_test "unit")

  add_executable(rag_chunking_test tests/unit/rag_chunking_test.cc)
  pomai_setup_test(rag_chunking_test)
  pomai_add_labeled_test(rag_chunking_test "unit")

  add_executable(rag_embedding_test tests/unit/rag_embedding_test.cc)
  pomai_setup_test(rag_embedding_test)
  pomai_add_labeled_test(rag_embedding_test "unit")

  add_executable(query_orchestrator_test tests/unit/query_orchestrator_test.cc)
  pomai_setup_test(query_orchestrator_test)
  pomai_add_labeled_test(query_orchestrator_test "unit")

  add_executable(text_membrane_test tests/unit/text_membrane_test.cc)
  pomai_setup_test(text_membrane_test)
  pomai_add_labeled_test(text_membrane_test "unit")

  add_executable(low_ram_limits_test tests/unit/low_ram_limits_test.cc)
  pomai_setup_test(low_ram_limits_test)
  pomai_add_labeled_test(low_ram_limits_test "unit")

  add_executable(edge_profile_options_test tests/unit/edge_profile_options_test.cc)
  pomai_setup_test(edge_profile_options_test)
  pomai_add_labeled_test(edge_profile_options_test "unit")

  add_executable(new_membrane_typed_api_test tests/unit/new_membrane_typed_api_test.cc)
  pomai_setup_test(new_membrane_typed_api_test)
  pomai_add_labeled_test(new_membrane_typed_api_test "unit")

  add_executable(membrane_capabilities_test tests/unit/membrane_capabilities_test.cc)
  pomai_setup_test(membrane_capabilities_test)
  pomai_add_labeled_test(membrane_capabilities_test "unit")

  add_executable(simd_new_membranes_test tests/unit/simd_new_membranes_test.cc)
  pomai_setup_test(simd_new_membranes_test)
  pomai_add_labeled_test(simd_new_membranes_test "unit")

  add_executable(no_train_dispatch_test tests/unit/no_train_dispatch_test.cc)
  pomai_setup_test(no_train_dispatch_test)
  pomai_add_labeled_test(no_train_dispatch_test "unit")

  add_executable(edge_ai_core_enhancements_test tests/unit/edge_ai_core_enhancements_test.cc)
  pomai_setup_test(edge_ai_core_enhancements_test)
  pomai_add_labeled_test(edge_ai_core_enhancements_test "unit")

  add_executable(mesh_lod_manager_test tests/unit/mesh_lod_manager_test.cc)
  pomai_setup_test(mesh_lod_manager_test)
  pomai_add_labeled_test(mesh_lod_manager_test "unit")

  add_executable(vulkan_memory_bridge_test tests/unit/vulkan_memory_bridge_test.cc)
  pomai_setup_test(vulkan_memory_bridge_test)
  pomai_add_labeled_test(vulkan_memory_bridge_test "unit")

  add_executable(recall_test tests/recall/recall_test.cc tests/recall/recall_dataset.cc tests/common/bruteforce_oracle.cc)
  pomai_setup_test(recall_test)
  pomai_add_labeled_test(recall_test "recall")

  # ---- Integration (embedded DB API) ----
  add_executable(db_basic_test tests/integ/db_basic_test.cc)
  pomai_setup_test(db_basic_test)
  pomai_add_labeled_test(db_basic_test "integ")

  add_executable(db_batch_test tests/integ/db_batch_test.cc)
  pomai_setup_test(db_batch_test)
  pomai_add_labeled_test(db_batch_test "integ")

  add_executable(db_persistence_test tests/integ/db_persistence_test.cc)
  pomai_setup_test(db_persistence_test)
  pomai_add_labeled_test(db_persistence_test "integ")

  add_executable(db_segment_test tests/integ/db_segment_test.cc)
  pomai_setup_test(db_segment_test)
  pomai_add_labeled_test(db_segment_test "integ")

  add_executable(db_open_test tests/integ/db_open_test.cc)
  pomai_setup_test(db_open_test)
  pomai_add_labeled_test(db_open_test "integ")
  
  add_executable(db_consistency_test tests/integ/db_consistency_test.cc)
  pomai_setup_test(db_consistency_test)
  pomai_add_labeled_test(db_consistency_test "integ")

  add_executable(db_partial_search_test tests/integ/db_partial_search_test.cc)
  pomai_setup_test(db_partial_search_test)
  pomai_add_labeled_test(db_partial_search_test "integ")

  add_executable(db_backpressure_test tests/integ/db_backpressure_test.cc)
  pomai_setup_test(db_backpressure_test)
  pomai_add_labeled_test(db_backpressure_test "integ")

  add_executable(db_edge_workload_test tests/integ/db_edge_workload_test.cc)
  pomai_setup_test(db_edge_workload_test)
  pomai_add_labeled_test(db_edge_workload_test "integ")

  add_executable(db_soak_stress_test tests/integ/db_soak_stress_test.cc)
  pomai_setup_test(db_soak_stress_test)
  pomai_add_labeled_test(db_soak_stress_test "stress")
  set_tests_properties(db_soak_stress_test PROPERTIES TIMEOUT 180)

  add_executable(db_error_paths_test tests/integ/db_error_paths_test.cc)
  pomai_setup_test(db_error_paths_test)
  pomai_add_labeled_test(db_error_paths_test "integ")

  add_executable(agent_memory_basic_test tests/integ/agent_memory_basic_test.cc)
  pomai_setup_test(agent_memory_basic_test)
  # Temporarily exclude from default CTest runs until AgentMemory persistence behavior is finalized.
  # pomai_add_labeled_test(agent_memory_basic_test "integ")

  add_executable(agent_memory_capi_test tests/integ/agent_memory_capi_test.cc)
  pomai_setup_test(agent_memory_capi_test)
  # Temporarily exclude from default CTest runs until AgentMemory C API is fully stabilized.
  # pomai_add_labeled_test(agent_memory_capi_test "integ")

  add_executable(routing_engine_test tests/integ/routing_engine_test.cc)
  pomai_setup_test(routing_engine_test)
  pomai_add_labeled_test(routing_engine_test "integ")

  add_executable(manifest_test tests/unit/manifest_test.cc)
  pomai_setup_test(manifest_test)
  pomai_add_labeled_test(manifest_test "unit")

  add_executable(multi_modal_test tests/unit/multi_modal_test.cc)
  pomai_setup_test(multi_modal_test)
  pomai_add_labeled_test(multi_modal_test "unit")

  add_executable(analytical_engine_test src/ai/analytical_engine_test.cc)
  pomai_setup_test(analytical_engine_test)
  pomai_add_labeled_test(analytical_engine_test "unit")

  add_executable(membrane_test tests/integ/membrane_test.cc)
  pomai_setup_test(membrane_test)
  pomai_add_labeled_test(membrane_test "integ")

  add_executable(membrane_persistence_test tests/integ/membrane_persistence_test.cc)
  pomai_setup_test(membrane_persistence_test)
  pomai_add_labeled_test(membrane_persistence_test "integ")

  add_executable(new_membrane_persistence_test tests/integ/new_membrane_persistence_test.cc)
  pomai_setup_test(new_membrane_persistence_test)
  pomai_add_labeled_test(new_membrane_persistence_test "integ")

  add_executable(simd_new_membranes_persistence_test tests/integ/simd_new_membranes_persistence_test.cc)
  pomai_setup_test(simd_new_membranes_persistence_test)
  pomai_add_labeled_test(simd_new_membranes_persistence_test "integ")

  add_executable(shard_concurrency_test tests/integ/shard_concurrency_test.cc)
  pomai_setup_test(shard_concurrency_test)
  pomai_add_labeled_test(shard_concurrency_test "integ")

  add_executable(iterator_test tests/integ/iterator_test.cc)
  pomai_setup_test(iterator_test)
  pomai_add_labeled_test(iterator_test "integ")

  add_executable(metadata_test tests/integ/metadata_test.cc)
  pomai_setup_test(metadata_test)
  pomai_add_labeled_test(metadata_test "integ")

  add_executable(search_filter_test tests/integ/search_filter_test.cc)
  pomai_setup_test(search_filter_test)
  pomai_add_labeled_test(search_filter_test "integ")

  add_executable(rag_contract_test tests/integ/rag_contract_test.cc)
  pomai_setup_test(rag_contract_test)
  pomai_add_labeled_test(rag_contract_test "integ")

  add_executable(rag_search_test tests/integ/rag_search_test.cc)
  pomai_setup_test(rag_search_test)
  pomai_add_labeled_test(rag_search_test "integ")

  add_executable(rag_pipeline_test tests/integ/rag_pipeline_test.cc)
  pomai_setup_test(rag_pipeline_test)
  pomai_add_labeled_test(rag_pipeline_test "integ")

  add_executable(search_newest_wins_test tests/integ/search_newest_wins_test.cc)
  pomai_setup_test(search_newest_wins_test)
  pomai_add_labeled_test(search_newest_wins_test "integ")

  add_executable(manifest_corruption_test tests/crash/manifest_corruption_test.cc)
  pomai_setup_test(manifest_corruption_test)
  pomai_add_labeled_test(manifest_corruption_test "crash")

  add_executable(sd_fault_injection_test tests/crash/sd_fault_injection_test.cc)
  pomai_setup_test(sd_fault_injection_test)
  pomai_add_labeled_test(sd_fault_injection_test "crash")

  add_executable(new_membrane_fault_test tests/crash/new_membrane_fault_test.cc)
  pomai_setup_test(new_membrane_fault_test)
  pomai_add_labeled_test(new_membrane_fault_test "crash")

  add_executable(simd_new_membranes_fault_test tests/crash/simd_new_membranes_fault_test.cc)
  pomai_setup_test(simd_new_membranes_fault_test)
  pomai_add_labeled_test(simd_new_membranes_fault_test "crash")

  add_executable(recovery_test tests/crash/recovery_test.cc)
  pomai_setup_test(recovery_test)
  pomai_add_labeled_test(recovery_test "crash")
  set_tests_properties(recovery_test PROPERTIES TIMEOUT 90)

  # ---- Crash (Gate #2) ----
  if (POMAI_ENABLE_CRASH_TESTS)
    add_executable(pomai_crash_test tests/crash/crash_replay_test.cc)
    
    # Manual setup loosely based on pomai_setup_test but without test_runner.cc
    target_link_libraries(pomai_crash_test PRIVATE pomai)
    if (MSVC)
      target_compile_options(pomai_crash_test PRIVATE /W4 /permissive-)
    else()
      target_compile_options(pomai_crash_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    target_include_directories(pomai_crash_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    if (POMAI_IS_TSAN)
      target_compile_options(pomai_crash_test PRIVATE -fno-pie)
      target_link_options(pomai_crash_test PRIVATE -no-pie)
    endif()

    add_test(NAME pomai_crash_replay COMMAND pomai_crash_test)
    set_tests_properties(pomai_crash_replay PROPERTIES LABELS "crash" TIMEOUT 60)

    add_executable(power_loss_stress_test tests/crash/power_loss_stress_test.cc)
    target_link_libraries(power_loss_stress_test PRIVATE pomai)
    if (MSVC)
      target_compile_options(power_loss_stress_test PRIVATE /W4 /permissive-)
    else()
      target_compile_options(power_loss_stress_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    target_include_directories(power_loss_stress_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    if (POMAI_IS_TSAN)
      target_compile_options(power_loss_stress_test PRIVATE -fno-pie)
      target_link_options(power_loss_stress_test PRIVATE -no-pie)
    endif()

    add_test(NAME power_loss_stress COMMAND power_loss_stress_test)
    set_tests_properties(power_loss_stress PROPERTIES LABELS "crash;stress" TIMEOUT 120)
  endif()

  # ---- TSAN (Gate #3) ----
  add_executable(db_concurrency_tsan_test tests/tsan/db_concurrency_tsan_test.cc)
  pomai_setup_test(db_concurrency_tsan_test)
  pomai_add_labeled_test(db_concurrency_tsan_test "tsan")

  add_executable(shard_runtime_tsan_test tests/tsan/shard_runtime_tsan_test.cc)
  pomai_setup_test(shard_runtime_tsan_test)
  pomai_add_labeled_test(shard_runtime_tsan_test "tsan")

  add_executable(backpressure_tsan_test tests/tsan/backpressure_tsan_test.cc)
  pomai_setup_test(backpressure_tsan_test)
  pomai_add_labeled_test(backpressure_tsan_test "tsan")

  add_executable(basic_workload_tsan_test tests/tsan/basic_workload_tsan_test.cc)
  pomai_setup_test(basic_workload_tsan_test)
  pomai_add_labeled_test(basic_workload_tsan_test "tsan")

  add_subdirectory(tests/fuzz)

  # Policy check: owned PomaiDB code must avoid raw malloc/new.
  # This runs as a normal ctest so CI catches allocator regressions.
  if (UNIX)
    add_test(
      NAME allocator_policy_test
      COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_no_malloc_new.sh" "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(allocator_policy_test PROPERTIES LABELS "unit;policy")
  endif()
endif()

# =========================
# Benchmarks
# =========================
if (POMAI_BUILD_BENCH)
  # Add benchmarks here later, use a similar helper to tests (but WITHOUT test_runner.cc)
  # Example:
  # add_executable(my_bench benchmarks/my_bench.cc)
  # target_link_libraries(my_bench PRIVATE pomai)
  # target_include_directories(my_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

set(POMAI_EXE_DEPS pomai)

# Comprehensive Benchmark (Industry-standard metrics)
add_executable(comprehensive_bench benchmarks/comprehensive_bench.cc)
target_link_libraries(comprehensive_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(comprehensive_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Ingestion Throughput Benchmark
add_executable(ingestion_bench benchmarks/ingestion_bench.cc)
target_link_libraries(ingestion_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(ingestion_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# RAG Benchmark
add_executable(rag_bench benchmarks/rag_bench.cc)
target_link_libraries(rag_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(rag_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


# CBR-S benchmark suite
add_executable(bench_cbrs benchmarks/bench_cbrs.cpp)
target_link_libraries(bench_cbrs PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(bench_cbrs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(bench_cbrs PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Deterministic CI benchmark for perf gate
add_executable(ci_perf_bench benchmarks/ci_perf_bench.cc)
target_link_libraries(ci_perf_bench PRIVATE ${POMAI_EXE_DEPS})

# GraphRAG Benchmark
add_executable(graph_bench benchmarks/graph_bench.cc)
target_link_libraries(graph_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(graph_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ci_perf_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Quantization Comparison Benchmark
add_executable(quantization_bench benchmarks/quantization_comp_bench.cc)
target_link_libraries(quantization_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(quantization_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# benchmark_a: Multi-Environment Stress Test (IoT / Edge / Cloud validation)
add_executable(benchmark_a benchmarks/palloc_env_stress.cc)
target_link_libraries(benchmark_a PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(benchmark_a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Encryption overhead benchmark harness
add_executable(encryption_perf_bench benchmarks/encryption_perf_bench.cc)
target_link_libraries(encryption_perf_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(encryption_perf_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(hybrid_orchestrator_bench benchmarks/hybrid_orchestrator_bench.cc)
target_link_libraries(hybrid_orchestrator_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(hybrid_orchestrator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(low_ram_profile_bench benchmarks/low_ram_profile_bench.cc)
target_link_libraries(low_ram_profile_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(low_ram_profile_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(new_membrane_bench benchmarks/new_membrane_bench.cc)
target_link_libraries(new_membrane_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(new_membrane_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(simd_new_membranes_bench benchmarks/simd_new_membranes_bench.cc)
target_link_libraries(simd_new_membranes_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(simd_new_membranes_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(edge_ai_core_bench benchmarks/edge_ai_core_bench.cc)
target_link_libraries(edge_ai_core_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(edge_ai_core_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(mesh_lod_bench benchmarks/mesh_lod_bench.cc)
target_link_libraries(mesh_lod_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(mesh_lod_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(vulkan_transfer_bench benchmarks/vulkan_transfer_bench.cc)
target_link_libraries(vulkan_transfer_bench PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(vulkan_transfer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Bounded benchmark smoke tests (avoid ctest default 1500s timeout on heavy workloads).
if (POMAI_REGISTER_BENCH_CTEST)
  set(POMAI_BENCH_TIMEOUT_LONG 600)
  set(POMAI_BENCH_TIMEOUT_MED 300)
  set(POMAI_BENCH_TIMEOUT_SHORT 180)

  add_test(NAME bench_comprehensive COMMAND $<TARGET_FILE:comprehensive_bench> --dataset small)
  set_tests_properties(bench_comprehensive PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  add_test(NAME bench_ingestion COMMAND $<TARGET_FILE:ingestion_bench> 10000 128)
  set_tests_properties(bench_ingestion PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_MED})

  add_test(NAME bench_rag COMMAND $<TARGET_FILE:rag_bench> 100 64 32)
  set_tests_properties(bench_rag PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_MED})

  add_test(NAME bench_ci_perf COMMAND $<TARGET_FILE:ci_perf_bench>)
  set_tests_properties(bench_ci_perf PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_graph COMMAND $<TARGET_FILE:graph_bench> 2000 1000 4000)
  set_tests_properties(bench_graph PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_MED})

  add_test(NAME bench_quantization COMMAND $<TARGET_FILE:quantization_bench>)
  set_tests_properties(bench_quantization PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_cbrs_smoke COMMAND $<TARGET_FILE:bench_cbrs> --n 8000 --queries 400 --topk 10)
  set_tests_properties(bench_cbrs_smoke PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  add_test(NAME bench_benchmark_a COMMAND ${CMAKE_COMMAND} -E env POMAI_BENCH_LOW_MEMORY=1 $<TARGET_FILE:benchmark_a>)
  set_tests_properties(bench_benchmark_a PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_LONG})

  add_test(NAME bench_encryption_perf COMMAND $<TARGET_FILE:encryption_perf_bench>)
  set_tests_properties(bench_encryption_perf PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_hybrid_orchestrator COMMAND $<TARGET_FILE:hybrid_orchestrator_bench>)
  set_tests_properties(bench_hybrid_orchestrator PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_low_ram_profile COMMAND $<TARGET_FILE:low_ram_profile_bench>)
  set_tests_properties(bench_low_ram_profile PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_new_membrane COMMAND $<TARGET_FILE:new_membrane_bench>)
  set_tests_properties(bench_new_membrane PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_simd_new_membranes COMMAND $<TARGET_FILE:simd_new_membranes_bench>)
  set_tests_properties(bench_simd_new_membranes PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_edge_ai_core COMMAND $<TARGET_FILE:edge_ai_core_bench>)
  set_tests_properties(bench_edge_ai_core PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_mesh_lod COMMAND $<TARGET_FILE:mesh_lod_bench>)
  set_tests_properties(bench_mesh_lod PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})

  add_test(NAME bench_vulkan_transfer COMMAND $<TARGET_FILE:vulkan_transfer_bench>)
  set_tests_properties(bench_vulkan_transfer PROPERTIES LABELS "bench" TIMEOUT ${POMAI_BENCH_TIMEOUT_SHORT})
endif()

# =========================
# Tools
# =========================
add_executable(pomai_inspect src/core/inspect.cc)
target_link_libraries(pomai_inspect PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(pomai_inspect PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Long-running gateway for Docker / VM (HTTP + ingest TCP); library use stays embedded.
add_executable(pomaidb_server tools/pomaidb_server.cc)
target_link_libraries(pomaidb_server PRIVATE ${POMAI_EXE_DEPS})
target_include_directories(pomaidb_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (MSVC)
  target_compile_options(pomaidb_server PRIVATE /W4 /permissive-)
else()
  target_compile_options(pomaidb_server PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =========================
# C API Tests & Examples
# =========================
if (POMAI_BUILD_TESTS)
  add_executable(c_api_test tests/test_c_api_basic.cpp)
  target_link_libraries(c_api_test PRIVATE pomai_c_static)
  target_sources(c_api_test PRIVATE tests/common/test_runner.cc)
  target_include_directories(c_api_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME c_api_test COMMAND c_api_test)

  add_executable(accuracy_test tests/accuracy_test.cc)
  target_link_libraries(accuracy_test PRIVATE pomai_c_static)
  target_sources(accuracy_test PRIVATE tests/common/test_runner.cc)
  target_include_directories(accuracy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME accuracy_test COMMAND accuracy_test)
endif()

add_executable(c_basic_example examples/c_basic.c)
target_link_libraries(c_basic_example PRIVATE pomai_c)
target_include_directories(c_basic_example PRIVATE include)

add_executable(c_scan_export_example examples/c_scan_export.c)
target_link_libraries(c_scan_export_example PRIVATE pomai_c)
target_include_directories(c_scan_export_example PRIVATE include)

add_executable(pomai_multimodal_demo examples/pomai_multimodal_demo.cpp)
target_link_libraries(pomai_multimodal_demo PRIVATE pomai)
target_include_directories(pomai_multimodal_demo PRIVATE include)

install(TARGETS pomai pomai_c pomai_c_static
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES
  include/pomai/c_api.h
  include/pomai/c_status.h
  include/pomai/c_types.h
  include/pomai/c_version.h
  include/pomai/database.h
  include/pomai/membrane_capabilities.h
  include/pomai/membrane_iterator.h
  DESTINATION include/pomai)
