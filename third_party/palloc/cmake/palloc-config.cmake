include(${CMAKE_CURRENT_LIST_DIR}/palloc.cmake)
get_filename_component(PALLOC_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" PATH)  # one up from the cmake dir, e.g. /usr/local/lib/cmake/palloc-2.0
get_filename_component(PALLOC_VERSION_DIR "${CMAKE_CURRENT_LIST_DIR}" NAME)
string(REPLACE "/lib/cmake" "/lib" PALLOC_LIBRARY_DIR "${PALLOC_CMAKE_DIR}")
if("${PALLOC_VERSION_DIR}" EQUAL "palloc")
  # top level install
  string(REPLACE "/lib/cmake" "/include" PALLOC_INCLUDE_DIR "${PALLOC_CMAKE_DIR}")
  set(PALLOC_OBJECT_DIR "${PALLOC_LIBRARY_DIR}")
else()
  # versioned
  string(REPLACE "/lib/cmake/" "/include/" PALLOC_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")
  string(REPLACE "/lib/cmake/" "/lib/" PALLOC_OBJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()
set(PALLOC_TARGET_DIR "${PALLOC_LIBRARY_DIR}") # legacy
