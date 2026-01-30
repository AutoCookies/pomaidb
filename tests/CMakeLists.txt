include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v2.13.10
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)

add_executable(pomai_tests
    common/test_main.cpp
    common/test_utils.cpp
    core/test_seed.cpp
    core/test_shard.cpp
    core/test_membrane.cpp
    core/test_orbit_index.cpp
    core/test_filtering.cpp
    core/test_filtering_correctness.cpp
    core/test_quality_filter_contract.cpp
    core/test_search.cpp
    core/test_scan.cpp
    storage/test_wal.cpp
    storage/test_wal_replay.cpp
    storage/test_checkpoint_recovery.cpp
    storage/test_corruption_detection.cpp
    concurrency/test_search_concurrency.cpp
    concurrency/test_ingest_concurrency.cpp
    concurrency/test_shutdown_races.cpp
    perf_guard/test_no_alloc_rerank.cpp
    perf_guard/test_latency_budget_logic.cpp
    tools/test_pomai_verify.cpp
    tools/test_pomai_dump.cpp
)

target_include_directories(pomai_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(pomai_tests PRIVATE
    pomai_api
    pomai_core
    pomai_storage
    pomai_index
    pomai_util
    Catch2::Catch2
)

include(Catch)
catch_discover_tests(pomai_tests)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS pomai_tests
)

add_custom_target(check_long
    COMMAND ${CMAKE_CTEST_COMMAND} -L long --output-on-failure
    DEPENDS pomai_tests
)
