_find_package(${ARGS})

if(CMAKE_CURRENT_LIST_DIR STREQUAL "${PALLOC_CMAKE_DIR}/${PALLOC_VERSION_DIR}")
    set(PALLOC_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
    # As in vcpkg.cmake
    if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
        set(PALLOC_LIBRARY_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib")
    else()
        set(PALLOC_LIBRARY_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")
    endif()
    set(PALLOC_OBJECT_DIR "${PALLOC_LIBRARY_DIR}")
    set(PALLOC_TARGET_DIR "${PALLOC_LIBRARY_DIR}")
endif()

# vcpkg always configures either a static or dynamic library.
# ensure to always expose the palloc target as either the static or dynamic build.
if(TARGET palloc-static AND NOT TARGET palloc)
  add_library(palloc INTERFACE IMPORTED)
  set_target_properties(palloc PROPERTIES INTERFACE_LINK_LIBRARIES palloc-static)
endif()
