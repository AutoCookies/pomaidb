cmake_minimum_required(VERSION 3.18)
project(libpalloc C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(PALLOC_SECURE            "Use full security mitigations (like guard pages, allocation randomization, double-free mitigation, and free-list corruption detection)" OFF)
option(PALLOC_PADDING           "Enable padding to detect heap block overflow (always on in DEBUG or SECURE mode, or with Valgrind/ASAN)" OFF)
option(PALLOC_OVERRIDE          "Override the standard malloc interface (i.e. define entry points for 'malloc', 'free', etc)" ON)
option(PALLOC_XMALLOC           "Enable abort() call on memory allocation failure by default" OFF)
option(PALLOC_SHOW_ERRORS       "Show error and warning messages by default (only enabled by default in DEBUG mode)" OFF)
option(PALLOC_GUARDED           "Build with guard pages behind certain object allocations (implies PALLOC_NO_PADDING=ON)" OFF)
option(PALLOC_USE_CXX           "Use the C++ compiler to compile the library (instead of the C compiler)" OFF)
option(PALLOC_OPT_ARCH          "Only for optimized builds: turn on architecture specific optimizations (for arm64: '-march=armv8.1-a' (2016))" OFF)
option(PALLOC_SEE_ASM           "Generate assembly files" OFF)
option(PALLOC_OSX_INTERPOSE     "Use interpose to override standard malloc on macOS" ON)
option(PALLOC_OSX_ZONE          "Use malloc zone to override standard malloc on macOS" ON)
option(PALLOC_WIN_REDIRECT      "Use redirection module ('palloc-redirect') on Windows if compiling palloc as a DLL" ON)
option(PALLOC_WIN_USE_FIXED_TLS "Use a fixed TLS slot on Windows to avoid extra tests in the malloc fast path" OFF)
option(PALLOC_LOCAL_DYNAMIC_TLS "Use local-dynamic-tls, a slightly slower but dlopen-compatible thread local storage mechanism (Unix)" OFF)
option(PALLOC_LIBC_MUSL         "Enable this when linking with musl libc" OFF)

option(PALLOC_DEBUG             "Enable assertion checks (enabled by default in a debug build)" OFF)
option(PALLOC_DEBUG_INTERNAL    "Enable assertion and internal invariant checks (enabled by default in a debug build)" OFF)
option(PALLOC_DEBUG_FULL        "Enable assertion checks and expensive internal heap invariant checking" OFF)

option(PALLOC_DEBUG_TSAN        "Build with thread sanitizer (needs clang)" OFF)
option(PALLOC_DEBUG_UBSAN       "Build with undefined-behavior sanitizer (needs clang++)" OFF)
option(PALLOC_TRACK_VALGRIND    "Compile with Valgrind support (adds a small overhead)" OFF)
option(PALLOC_TRACK_ASAN        "Compile with address sanitizer support (adds a small overhead)" OFF)
option(PALLOC_TRACK_ETW         "Compile with Windows event tracing (ETW) support (adds a small overhead)" OFF)

option(PALLOC_BUILD_SHARED      "Build shared library" ON)
option(PALLOC_BUILD_STATIC      "Build static library" ON)
option(PALLOC_BUILD_OBJECT      "Build object library" ON)
option(PALLOC_BUILD_TESTS       "Build test executables" ON)

option(PALLOC_SKIP_COLLECT_ON_EXIT "Skip collecting memory on program exit" OFF)
option(PALLOC_NO_PADDING        "Force no use of padding even in DEBUG mode etc." OFF)
option(PALLOC_INSTALL_TOPLEVEL  "Install directly into $CMAKE_INSTALL_PREFIX instead of PREFIX/lib/palloc-version" OFF)
option(PALLOC_NO_THP            "Disable transparent huge pages support on Linux/Android for the palloc process only" OFF)
option(PALLOC_EXTRA_CPPDEFS     "Extra pre-processor definitions (use as `-DMI_EXTRA_CPPDEFS=\"opt1=val1;opt2=val2\"`)" "")

# negated options for vcpkg features
option(PALLOC_NO_USE_CXX        "Use plain C compilation (has priority over PALLOC_USE_CXX)" OFF)
option(PALLOC_NO_OPT_ARCH       "Do not use architecture specific optimizations (like '-march=armv8.1-a' for example) (has priority over PALLOC_OPT_ARCH)" OFF)

# deprecated options
option(PALLOC_WIN_USE_FLS       "Use Fiber local storage on Windows to detect thread termination (deprecated)" OFF)
option(PALLOC_CHECK_FULL        "Use full internal invariant checking in DEBUG mode (deprecated, use PALLOC_DEBUG_FULL instead)" OFF)
option(PALLOC_USE_LIBATOMIC     "Explicitly link with -latomic (on older systems) (deprecated and detected automatically)" OFF)

include(CheckLinkerFlag)    # requires cmake 3.18
include(CheckIncludeFiles)
include(GNUInstallDirs)
include("cmake/palloc-config-version.cmake")

set(palloc_sources
    src/alloc.c
    src/alloc-aligned.c
    src/alloc-posix.c
    src/arena.c
    src/bitmap.c
    src/heap.c
    src/init.c
    src/libc.c
    src/options.c
    src/os.c
    src/page.c
    src/random.c
    src/segment.c
    src/segment-map.c
    src/stats.c
    src/prim/prim.c)

set(palloc_cflags "")
set(palloc_cflags_static "")            # extra flags for a static library build
set(palloc_cflags_dynamic "")           # extra flags for a shared-object library build
set(palloc_libraries "")

if(PALLOC_EXTRA_CPPDEFS)
 set(palloc_defines ${PALLOC_EXTRA_CPPDEFS})
else()
 set(palloc_defines "")
endif()

# pass git revision as a define
if(EXISTS "${CMAKE_SOURCE_DIR}/.git/index")
  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} "describe" OUTPUT_VARIABLE palloc_git_describe RESULT_VARIABLE palloc_git_res ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(palloc_git_res EQUAL "0")
      list(APPEND palloc_defines "PALLOC_GIT_DESCRIBE=${palloc_git_describe}")
      # add to dependencies so we rebuild if the git head commit changes
      set_property(GLOBAL APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/index")
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# Convenience: set default build type and compiler depending on the build directory
# -----------------------------------------------------------------------------

message(STATUS "")
if (NOT CMAKE_BUILD_TYPE)
  if ("${CMAKE_BINARY_DIR}" MATCHES ".*((D|d)ebug|asan|tsan|ubsan|valgrind)$")
    message(STATUS "No build type selected, default to 'Debug'")
    set(CMAKE_BUILD_TYPE "Debug")
  else()
    message(STATUS "No build type selected, default to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
  endif()
endif()

if (CMAKE_GENERATOR MATCHES "^Visual Studio.*$")
  message(STATUS "Note: when building with Visual Studio the build type is specified when building.")
  message(STATUS "For example: 'cmake --build . --config=Release")
endif()

if("${CMAKE_BINARY_DIR}" MATCHES ".*(S|s)ecure$")
  message(STATUS "Default to secure build")
  set(PALLOC_SECURE "ON")
endif()


# Determine architecture
set(PALLOC_OPT_ARCH_FLAGS "")
set(PALLOC_ARCH "unknown")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|i[3456]86)$" OR CMAKE_GENERATOR_PLATFORM MATCHES "^(x86|Win32)$")
  set(PALLOC_ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|x64|amd64|AMD64)$" OR CMAKE_GENERATOR_PLATFORM STREQUAL "x64" OR "x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES) # must be before arm64
  set(PALLOC_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv[89].?|ARM64)$" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES)
  set(PALLOC_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv[34567].?|ARM)$")
  set(PALLOC_ARCH "arm32")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv|riscv32|riscv64)$")
  if(CMAKE_SIZEOF_VOID_P==4)
    set(PALLOC_ARCH "riscv32")
  else()
    set(PALLOC_ARCH "riscv64")
  endif()
else()
  set(PALLOC_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()
message(STATUS "Architecture: ${PALLOC_ARCH}") # (${CMAKE_SYSTEM_PROCESSOR}, ${CMAKE_GENERATOR_PLATFORM}, ${CMAKE_GENERATOR})")

# negative overrides (mainly to support vcpkg features)
if(PALLOC_NO_USE_CXX)
  set(PALLOC_USE_CXX "OFF")
endif()
if(PALLOC_NO_OPT_ARCH)
  set(PALLOC_OPT_ARCH "OFF")
elseif(PALLOC_ARCH STREQUAL "arm64")
  set(PALLOC_OPT_ARCH "ON")  # enable armv8.1-a by default on arm64 unless PALLOC_NO_OPT_ARCH is set
endif()


# -----------------------------------------------------------------------------
# Process options
# -----------------------------------------------------------------------------

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(PALLOC_CLANG_CL "ON")
endif()

# put -Wall early so other warnings can be disabled selectively
if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang")
  if (PALLOC_CLANG_CL)
    list(APPEND palloc_cflags -W)
  else()
    list(APPEND palloc_cflags -Wall -Wextra -Wpedantic)
  endif()
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    list(APPEND palloc_cflags -Wall -Wextra)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "Intel")
    list(APPEND palloc_cflags -Wall)
endif()

# force C++ compilation with msvc or clang-cl to use modern C++ atomics
if(CMAKE_C_COMPILER_ID MATCHES "MSVC|Intel" OR PALLOC_CLANG_CL)
  set(PALLOC_USE_CXX "ON")
endif()

if(PALLOC_OVERRIDE)
  message(STATUS "Override standard malloc (PALLOC_OVERRIDE=ON)")
  if(APPLE)
    if(PALLOC_OSX_ZONE)
      # use zone's on macOS
      message(STATUS "  Use malloc zone to override malloc (PALLOC_OSX_ZONE=ON)")
      list(APPEND palloc_sources src/prim/osx/alloc-override-zone.c)
      list(APPEND palloc_defines PALLOC_OSX_ZONE=1)
      if (NOT PALLOC_OSX_INTERPOSE)
        message(STATUS "  WARNING: zone overriding usually also needs interpose (use -DMI_OSX_INTERPOSE=ON)")
      endif()
    endif()
    if(PALLOC_OSX_INTERPOSE)
      # use interpose on macOS
      message(STATUS "  Use interpose to override malloc (PALLOC_OSX_INTERPOSE=ON)")
      list(APPEND palloc_defines PALLOC_OSX_INTERPOSE=1)
      if (NOT PALLOC_OSX_ZONE)
        message(STATUS "  WARNING: interpose usually also needs zone overriding (use -DMI_OSX_ZONE=ON)")
      endif()
    endif()
    if(PALLOC_USE_CXX AND PALLOC_OSX_INTERPOSE)
      message(STATUS "  WARNING: if dynamically overriding malloc/free, it is more reliable to build palloc as C code (use -DMI_USE_CXX=OFF)")
    endif()
  endif()
endif()

if(WIN32)
  if (NOT PALLOC_WIN_REDIRECT)
    # use a negative define for backward compatibility
    list(APPEND palloc_defines PALLOC_WIN_NOREDIRECT=1)
  endif()
endif()

if(PALLOC_SECURE)
  message(STATUS "Set full secure build (PALLOC_SECURE=ON)")
  list(APPEND palloc_defines PALLOC_SECURE=4)
endif()

if(PALLOC_TRACK_VALGRIND)
  CHECK_INCLUDE_FILES("valgrind/valgrind.h;valgrind/memcheck.h" PALLOC_HAS_VALGRINDH)
  if (NOT PALLOC_HAS_VALGRINDH)
    set(PALLOC_TRACK_VALGRIND OFF)
    message(WARNING "Cannot find the 'valgrind/valgrind.h' and 'valgrind/memcheck.h' -- install valgrind first?")
    message(STATUS  "Disabling Valgrind support (PALLOC_TRACK_VALGRIND=OFF)")
  else()
    message(STATUS "Compile with Valgrind support (PALLOC_TRACK_VALGRIND=ON)")
    list(APPEND palloc_defines PALLOC_TRACK_VALGRIND=1)
  endif()
endif()

if(PALLOC_TRACK_ASAN)
  if (APPLE AND PALLOC_OVERRIDE)
    set(PALLOC_TRACK_ASAN OFF)
    message(WARNING "Cannot enable address sanitizer support on macOS if PALLOC_OVERRIDE is ON (PALLOC_TRACK_ASAN=OFF)")
  endif()
  if (PALLOC_TRACK_VALGRIND)
    set(PALLOC_TRACK_ASAN OFF)
    message(WARNING "Cannot enable address sanitizer support with also Valgrind support enabled (PALLOC_TRACK_ASAN=OFF)")
  endif()
  if(PALLOC_TRACK_ASAN)
    CHECK_INCLUDE_FILES("sanitizer/asan_interface.h" PALLOC_HAS_ASANH)
    if (NOT PALLOC_HAS_ASANH)
      set(PALLOC_TRACK_ASAN OFF)
      message(WARNING "Cannot find the 'sanitizer/asan_interface.h' -- install address sanitizer support first")
      message(STATUS  "Compile **without** address sanitizer support (PALLOC_TRACK_ASAN=OFF)")
    else()
      message(STATUS "Compile with address sanitizer support (PALLOC_TRACK_ASAN=ON)")
      list(APPEND palloc_defines PALLOC_TRACK_ASAN=1)
      list(APPEND palloc_cflags -fsanitize=address)
      list(APPEND palloc_libraries -fsanitize=address)
    endif()
  endif()
endif()

if(PALLOC_TRACK_ETW)
  if(NOT WIN32)
    set(PALLOC_TRACK_ETW OFF)
    message(WARNING "Can only enable ETW support on Windows (PALLOC_TRACK_ETW=OFF)")
  endif()
  if (PALLOC_TRACK_VALGRIND OR PALLOC_TRACK_ASAN)
    set(PALLOC_TRACK_ETW OFF)
    message(WARNING "Cannot enable ETW support with also Valgrind or ASAN support enabled (PALLOC_TRACK_ETW=OFF)")
  endif()
  if(PALLOC_TRACK_ETW)
    message(STATUS "Compile with Windows event tracing support (PALLOC_TRACK_ETW=ON)")
    list(APPEND palloc_defines PALLOC_TRACK_ETW=1)
  endif()
endif()

if(PALLOC_GUARDED)
  message(STATUS "Compile guard pages behind certain object allocations (PALLOC_GUARDED=ON)")
  list(APPEND palloc_defines PALLOC_GUARDED=1)
  if(NOT PALLOC_NO_PADDING)
    message(STATUS "  Disabling padding due to guard pages (PALLOC_NO_PADDING=ON)")
    set(PALLOC_NO_PADDING ON)
  endif()
endif()

if(PALLOC_SEE_ASM)
  message(STATUS "Generate assembly listings (PALLOC_SEE_ASM=ON)")
  list(APPEND palloc_cflags -save-temps)
  if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang")
    message(STATUS "No GNU Line marker")
    list(APPEND palloc_cflags -Wno-gnu-line-marker)
  endif()
endif()

if(PALLOC_CHECK_FULL)
  message(STATUS "The PALLOC_CHECK_FULL option is deprecated, use PALLOC_DEBUG_FULL instead")
  set(PALLOC_DEBUG_FULL "ON")
endif()

if (PALLOC_SKIP_COLLECT_ON_EXIT)
  message(STATUS "Skip collecting memory on program exit (PALLOC_SKIP_COLLECT_ON_EXIT=ON)")
  list(APPEND palloc_defines PALLOC_SKIP_COLLECT_ON_EXIT=1)
endif()

if(PALLOC_DEBUG_FULL)
  message(STATUS "Set debug level to full assertion and internal invariant checking (PALLOC_DEBUG_FULL=ON, expensive)")
  list(APPEND palloc_defines PALLOC_DEBUG=3)   # full invariant checking (palloc_assert, palloc_assert_internal, and palloc_assert_expensive)
elseif(PALLOC_DEBUG_INTERNAL)
  message(STATUS "Set debug level to internal assertion and invariant checking (PALLOC_DEBUG_INTERNAL=ON)")
  list(APPEND palloc_defines PALLOC_DEBUG=2)   # invariant checking (palloc_assert and palloc_assert_internal)
elseif(PALLOC_DEBUG)
  message(STATUS "Set debug level to assertion checking (PALLOC_DEBUG=ON)")
  list(APPEND palloc_defines PALLOC_DEBUG=1)   # assertion checking (palloc_assert)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(STATUS "Set debug level to internal assertion and invariant checking (CMAKE_BUILD_TYPE=Debug)")
  set(PALLOC_DEBUG_INTERNAL ON)
  list(APPEND palloc_defines PALLOC_DEBUG=2)   # invariant checking (palloc_assert and palloc_assert_internal)
endif()

if(PALLOC_NO_PADDING)
  message(STATUS "Suppress any padding of heap blocks (PALLOC_NO_PADDING=ON)")
  list(APPEND palloc_defines PALLOC_PADDING=0)
else()
  if(PALLOC_PADDING)
    message(STATUS "Enable explicit padding of heap blocks (PALLOC_PADDING=ON)")
    list(APPEND palloc_defines PALLOC_PADDING=1)
  endif()
endif()

if(PALLOC_XMALLOC)
  message(STATUS "Enable abort() calls on memory allocation failure (PALLOC_XMALLOC=ON)")
  list(APPEND palloc_defines PALLOC_XMALLOC=1)
endif()

if(PALLOC_SHOW_ERRORS)
  message(STATUS "Enable printing of error and warning messages by default (PALLOC_SHOW_ERRORS=ON)")
  list(APPEND palloc_defines PALLOC_SHOW_ERRORS=1)
endif()

if(PALLOC_DEBUG_TSAN)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(STATUS "Build with thread sanitizer (PALLOC_DEBUG_TSAN=ON)")
    list(APPEND palloc_defines PALLOC_TSAN=1)
    list(APPEND palloc_cflags -fsanitize=thread -g -O1)
    list(APPEND palloc_libraries -fsanitize=thread)
  else()
    message(WARNING "Can only use thread sanitizer with clang (PALLOC_DEBUG_TSAN=ON but ignored)")
  endif()
endif()

if(PALLOC_DEBUG_UBSAN)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(STATUS "Build with undefined-behavior sanitizer (PALLOC_DEBUG_UBSAN=ON)")
      list(APPEND palloc_defines PALLOC_UBSAN=1)
      list(APPEND palloc_cflags -fsanitize=undefined -g -fno-sanitize-recover=undefined)
      list(APPEND palloc_libraries -fsanitize=undefined)
      if (NOT PALLOC_USE_CXX)
        message(STATUS "(switch to use C++ due to PALLOC_DEBUG_UBSAN)")
        set(PALLOC_USE_CXX "ON")
      endif()
    else()
      message(WARNING "Can only use undefined-behavior sanitizer with clang++ (PALLOC_DEBUG_UBSAN=ON but ignored)")
    endif()
  else()
    message(WARNING "Can only use undefined-behavior sanitizer with a debug build (CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})")
  endif()
endif()

if(PALLOC_USE_CXX)
  message(STATUS "Use the C++ compiler to compile (PALLOC_USE_CXX=ON)")
  set_source_files_properties(${palloc_sources} PROPERTIES LANGUAGE CXX )
  set_source_files_properties(src/static.c test/test-api.c test/test-api-fill test/test-stress PROPERTIES LANGUAGE CXX )
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    list(APPEND palloc_cflags -Wno-deprecated)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    list(APPEND palloc_cflags -Kc++)
  endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  if(PALLOC_NO_THP)
    message(STATUS "Disable transparent huge pages support (PALLOC_NO_THP=ON)")
    list(APPEND palloc_defines PALLOC_NO_THP=1)
  endif()
endif()

if(PALLOC_LIBC_MUSL)
  message(STATUS "Assume using musl libc (PALLOC_LIBC_MUSL=ON)")
  list(APPEND palloc_defines PALLOC_LIBC_MUSL=1)
endif()

if(PALLOC_WIN_USE_FLS)
  message(STATUS "Use the Fiber API to detect thread termination (deprecated) (PALLOC_WIN_USE_FLS=ON)")
  list(APPEND palloc_defines PALLOC_WIN_USE_FLS=1)
endif()

if(PALLOC_WIN_USE_FIXED_TLS)
  message(STATUS "Use fixed TLS slot on Windows to avoid extra tests in the malloc fast path (PALLOC_WIN_USE_FIXED_TLS=ON)")
  list(APPEND palloc_defines PALLOC_WIN_USE_FIXED_TLS=1)
endif()

# Check /proc/cpuinfo for an SV39 MMU and limit the virtual address bits.
# (this will skip the aligned hinting in that case. Issue #939, #949)
if (EXISTS /proc/cpuinfo)
  file(STRINGS /proc/cpuinfo palloc_sv39_mmu REGEX "^mmu[ \t]+:[ \t]+sv39$")
  if (palloc_sv39_mmu)
    MESSAGE( STATUS "Set virtual address bits to 39 (SV39 MMU detected)" )
    list(APPEND palloc_defines PALLOC_DEFAULT_VIRTUAL_ADDRESS_BITS=39)
  endif()
endif()

# On Haiku use `-DCMAKE_INSTALL_PREFIX` instead, issue #788
# if(CMAKE_SYSTEM_NAME MATCHES "Haiku")
#   SET(CMAKE_INSTALL_LIBDIR ~/config/non-packaged/lib)
#   SET(CMAKE_INSTALL_INCLUDEDIR ~/config/non-packaged/headers)
# endif()

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU" AND NOT PALLOC_CLANG_CL)
  list(APPEND palloc_cflags -Wno-unknown-pragmas -fvisibility=hidden)
  if(NOT PALLOC_USE_CXX)
    list(APPEND palloc_cflags -Wstrict-prototypes)
  endif()
  if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang")
    list(APPEND palloc_cflags -Wno-static-in-inline)
  endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Intel")
  list(APPEND palloc_cflags -fvisibility=hidden)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU|Intel" AND NOT CMAKE_SYSTEM_NAME MATCHES "Haiku" AND NOT PALLOC_CLANG_CL)
  if(PALLOC_LOCAL_DYNAMIC_TLS)
    list(APPEND palloc_cflags -ftls-model=local-dynamic)
  else()
    if(PALLOC_LIBC_MUSL)
      # with musl we use local-dynamic for the static build, see issue #644
      list(APPEND palloc_cflags_static  -ftls-model=local-dynamic)
      list(APPEND palloc_cflags_dynamic -ftls-model=initial-exec)
      message(STATUS "Use local dynamic TLS for the static build (since PALLOC_LIBC_MUSL=ON)")
    else()
      list(APPEND palloc_cflags -ftls-model=initial-exec)
    endif()
  endif()
  if(PALLOC_OVERRIDE)
    list(APPEND palloc_cflags -fno-builtin-malloc)
  endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU|Intel" AND NOT CMAKE_SYSTEM_NAME MATCHES "Haiku")
  if(PALLOC_OPT_ARCH)
    if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang" AND CMAKE_OSX_ARCHITECTURES)   # to support multi-arch binaries (#999)
      if("arm64" IN_LIST CMAKE_OSX_ARCHITECTURES)
        list(APPEND PALLOC_OPT_ARCH_FLAGS "-Xarch_arm64;-march=armv8.1-a")
      endif()
    elseif(PALLOC_ARCH STREQUAL "arm64")
      set(PALLOC_OPT_ARCH_FLAGS "-march=armv8.1-a")  # fast atomics
    endif()
  endif()
endif()

if (MSVC AND MSVC_VERSION GREATER_EQUAL 1914)
  list(APPEND palloc_cflags /Zc:__cplusplus)
  if(PALLOC_OPT_ARCH AND NOT PALLOC_CLANG_CL)
    if(PALLOC_ARCH STREQUAL "arm64")
      set(PALLOC_OPT_ARCH_FLAGS "/arch:armv8.1")           # fast atomics
    endif()
  endif()
endif()

if(MINGW)
  add_definitions(-D_WIN32_WINNT=0x600)                # issue #976
endif()

if(PALLOC_OPT_ARCH_FLAGS)
  list(APPEND palloc_cflags ${PALLOC_OPT_ARCH_FLAGS})
  message(STATUS "Architecture specific optimization is enabled (with ${PALLOC_OPT_ARCH_FLAGS}) (PALLOC_OPT_ARCH=ON)")
endif()

# extra needed libraries

# we prefer -l<lib> test over `find_library` as sometimes core libraries
# like `libatomic` are not on the system path (see issue #898)
function(find_link_library libname outlibname)
  check_linker_flag(C "-l${libname}" palloc_has_lib${libname})
  if (palloc_has_lib${libname})
    message(VERBOSE "link library: -l${libname}")
    set(${outlibname} ${libname} PARENT_SCOPE)
  else()
    find_library(PALLOC_LIBPATH_${libname} ${libname})
    if (PALLOC_LIBPATH_${libname})
      message(VERBOSE "link library ${libname} at ${PALLOC_LIBPATH_${libname}}")
      set(${outlibname} ${PALLOC_LIBPATH_${libname}} PARENT_SCOPE)
    else()
      message(VERBOSE "link library not found: ${libname}")
      set(${outlibname} "" PARENT_SCOPE)
    endif()
  endif()
endfunction()

if(WIN32)
  list(APPEND palloc_libraries psapi shell32 user32 advapi32 bcrypt)
else()
  find_link_library("pthread" PALLOC_LIB_PTHREAD)
  if(PALLOC_LIB_PTHREAD)
    list(APPEND palloc_libraries "${PALLOC_LIB_PTHREAD}")
  endif()
  find_link_library("rt" PALLOC_LIB_RT)
  if(PALLOC_LIB_RT)
    list(APPEND palloc_libraries "${PALLOC_LIB_RT}")
  endif()
  find_link_library("atomic" PALLOC_LIB_ATOMIC)
  if(PALLOC_LIB_ATOMIC)
    list(APPEND palloc_libraries "${PALLOC_LIB_ATOMIC}")
  endif()
endif()


# -----------------------------------------------------------------------------
# Install and output names
# -----------------------------------------------------------------------------

# dynamic/shared library and symlinks always go to /usr/local/lib equivalent
# we use ${CMAKE_INSTALL_BINDIR} and ${CMAKE_INSTALL_LIBDIR}.

# static libraries and object files, includes, and cmake config files
# are either installed at top level, or use versioned directories for side-by-side installation (default)
if (PALLOC_INSTALL_TOPLEVEL)
  set(palloc_install_objdir     "${CMAKE_INSTALL_LIBDIR}")
  set(palloc_install_incdir     "${CMAKE_INSTALL_INCLUDEDIR}")
  set(palloc_install_cmakedir   "${CMAKE_INSTALL_LIBDIR}/cmake/palloc")
else()
  set(palloc_install_objdir     "${CMAKE_INSTALL_LIBDIR}/palloc-${palloc_version}")       # for static library and object files
  set(palloc_install_incdir     "${CMAKE_INSTALL_INCLUDEDIR}/palloc-${palloc_version}")   # for includes
  set(palloc_install_cmakedir   "${CMAKE_INSTALL_LIBDIR}/cmake/palloc-${palloc_version}") # for cmake package info
endif()

set(palloc_libname "palloc")
if(PALLOC_SECURE)
  set(palloc_libname "${palloc_libname}-secure")
endif()
if(PALLOC_TRACK_VALGRIND)
  set(palloc_libname "${palloc_libname}-valgrind")
endif()
if(PALLOC_TRACK_ASAN)
  set(palloc_libname "${palloc_libname}-asan")
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LC)
list(APPEND palloc_defines "PALLOC_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE_LC}")  #todo: multi-config project needs $<CONFIG> ?
if(CMAKE_BUILD_TYPE_LC MATCHES "^(release|relwithdebinfo|minsizerel|none)$")
  list(APPEND palloc_defines PALLOC_BUILD_RELEASE)
else()
  set(palloc_libname "${palloc_libname}-${CMAKE_BUILD_TYPE_LC}") #append build type (e.g. -debug) if not a release version
endif()

if(PALLOC_BUILD_SHARED)
  list(APPEND palloc_build_targets "shared")
endif()
if(PALLOC_BUILD_STATIC)
  list(APPEND palloc_build_targets "static")
endif()
if(PALLOC_BUILD_OBJECT)
  list(APPEND palloc_build_targets "object")
endif()
if(PALLOC_BUILD_TESTS)
  list(APPEND palloc_build_targets "tests")
endif()

message(STATUS "")
message(STATUS "Library name     : ${palloc_libname}")
message(STATUS "Version          : ${palloc_version}.${palloc_version_patch}")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE_LC}")
if(PALLOC_USE_CXX)
  message(STATUS "C++ Compiler     : ${CMAKE_CXX_COMPILER}")
else()
  message(STATUS "C Compiler       : ${CMAKE_C_COMPILER}")
endif()
message(STATUS "Compiler flags   : ${palloc_cflags}")
message(STATUS "Compiler defines : ${palloc_defines}")
message(STATUS "Link libraries   : ${palloc_libraries}")
message(STATUS "Build targets    : ${palloc_build_targets}")
message(STATUS "")

# -----------------------------------------------------------------------------
# Main targets
# -----------------------------------------------------------------------------

# shared library
if(PALLOC_BUILD_SHARED)
  add_library(palloc SHARED ${palloc_sources})
  set_target_properties(palloc PROPERTIES VERSION ${palloc_version} SOVERSION ${palloc_version_major} OUTPUT_NAME ${palloc_libname} )
  target_compile_definitions(palloc PRIVATE ${palloc_defines} PALLOC_SHARED_LIB PALLOC_SHARED_LIB_EXPORT)
  target_compile_options(palloc PRIVATE ${palloc_cflags} ${palloc_cflags_dynamic})
  target_link_libraries(palloc PRIVATE ${palloc_libraries})
  target_include_directories(palloc PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${palloc_install_incdir}>
  )
  install(TARGETS palloc EXPORT palloc ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT palloc DESTINATION ${palloc_install_cmakedir})

  if(WIN32 AND NOT MINGW)
    # On windows, the import library name for the dll would clash with the static palloc.lib library
    # so we postfix the dll import library with `.dll.lib` (and also the .pdb debug file)
    set_property(TARGET palloc PROPERTY ARCHIVE_OUTPUT_NAME "${palloc_libname}.dll" )
    install(FILES "$<TARGET_FILE_DIR:palloc>/${palloc_libname}.dll.lib" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    set_property(TARGET palloc PROPERTY PDB_NAME "${palloc_libname}.dll")
    # don't try to install the pdb since it may not be generated depending on the configuration
    # install(FILES "$<TARGET_FILE_DIR:palloc>/${palloc_libname}.dll.pdb" DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  if(WIN32 AND PALLOC_WIN_REDIRECT)
    if(MINGW)
      set_property(TARGET palloc PROPERTY PREFIX "")
    endif()
    # On windows, link and copy the palloc redirection dll too.
    if(CMAKE_GENERATOR_PLATFORM STREQUAL "arm64ec")
      set(PALLOC_REDIRECT_SUFFIX "-arm64ec")
    elseif(PALLOC_ARCH STREQUAL "x64")
      set(PALLOC_REDIRECT_SUFFIX "")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
        message(STATUS "Note: x64 code emulated on Windows for arm64 should use an arm64ec build of 'palloc.dll'")
        message(STATUS "      together with 'palloc-redirect-arm64ec.dll'. See the 'bin\\readme.md' for more information.")
      endif()
    elseif(PALLOC_ARCH STREQUAL "x86")
      set(PALLOC_REDIRECT_SUFFIX "32")
    else()
      set(PALLOC_REDIRECT_SUFFIX "-${PALLOC_ARCH}")  # -arm64 etc.
    endif()

    target_link_libraries(palloc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bin/palloc-redirect${PALLOC_REDIRECT_SUFFIX}.lib)  # the DLL import library
    add_custom_command(TARGET palloc POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/bin/palloc-redirect${PALLOC_REDIRECT_SUFFIX}.dll" $<TARGET_FILE_DIR:palloc>
      COMMENT "Copy palloc-redirect${PALLOC_REDIRECT_SUFFIX}.dll to output directory")
    install(FILES "$<TARGET_FILE_DIR:palloc>/palloc-redirect${PALLOC_REDIRECT_SUFFIX}.dll" DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()


# static library
if (PALLOC_BUILD_STATIC)
  add_library(palloc-static STATIC ${palloc_sources})
  set_property(TARGET palloc-static PROPERTY OUTPUT_NAME ${palloc_libname})
  set_property(TARGET palloc-static PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(palloc-static PRIVATE ${palloc_defines} PALLOC_STATIC_LIB)
  target_compile_options(palloc-static PRIVATE ${palloc_cflags} ${palloc_cflags_static})
  target_link_libraries(palloc-static PRIVATE ${palloc_libraries})
  target_include_directories(palloc-static PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${palloc_install_incdir}>
  )
  install(TARGETS palloc-static EXPORT palloc DESTINATION ${palloc_install_objdir} LIBRARY)
  install(EXPORT palloc DESTINATION ${palloc_install_cmakedir})
endif()

# install include files
install(FILES include/palloc.h DESTINATION ${palloc_install_incdir})
install(FILES include/palloc-override.h DESTINATION ${palloc_install_incdir})
install(FILES include/palloc-new-delete.h DESTINATION ${palloc_install_incdir})
install(FILES include/palloc-stats.h DESTINATION ${palloc_install_incdir})
install(FILES cmake/palloc-config.cmake DESTINATION ${palloc_install_cmakedir})
install(FILES cmake/palloc-config-version.cmake DESTINATION ${palloc_install_cmakedir})


# single object file for more predictable static overriding
if (PALLOC_BUILD_OBJECT)
  add_library(palloc-obj OBJECT src/static.c)
  set_property(TARGET palloc-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(palloc-obj PRIVATE ${palloc_defines})
  target_compile_options(palloc-obj PRIVATE ${palloc_cflags} ${palloc_cflags_static})
  target_include_directories(palloc-obj PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${palloc_install_incdir}>
  )

  # Copy the generated object file (`static.o`) to the output directory (as `palloc.o`)
  if(CMAKE_GENERATOR MATCHES "^Visual Studio.*$")
    set(palloc-obj-static "${CMAKE_CURRENT_BINARY_DIR}/palloc-obj.dir/$<CONFIG>/static${CMAKE_C_OUTPUT_EXTENSION}")
  else()
    set(palloc-obj-static "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/palloc-obj.dir/src/static.c${CMAKE_C_OUTPUT_EXTENSION}")
  endif()
  set(palloc-obj-out "${CMAKE_CURRENT_BINARY_DIR}/${palloc_libname}${CMAKE_C_OUTPUT_EXTENSION}")
  add_custom_command(OUTPUT ${palloc-obj-out} DEPENDS palloc-obj COMMAND "${CMAKE_COMMAND}" -E copy "${palloc-obj-static}" "${palloc-obj-out}")
  add_custom_target(palloc-obj-target ALL DEPENDS ${palloc-obj-out})


  # the following seems to lead to cmake warnings/errors on some systems, disable for now :-(
  # install(TARGETS palloc-obj EXPORT palloc DESTINATION ${palloc_install_objdir})

  # the FILES expression can also be: $<TARGET_OBJECTS:palloc-obj>
  # but that fails cmake versions less than 3.10 so we leave it as is for now
  install(FILES ${palloc-obj-static}
          DESTINATION ${palloc_install_objdir}
          RENAME ${palloc_libname}${CMAKE_C_OUTPUT_EXTENSION} )
endif()


# pkg-config file support
set(palloc_pc_libraries "")
foreach(item IN LISTS palloc_libraries)
  if(item MATCHES " *[-].*")
    set(palloc_pc_libraries "${palloc_pc_libraries} ${item}")
  else()
    set(palloc_pc_libraries "${palloc_pc_libraries} -l${item}")
  endif()
endforeach()

include("cmake/JoinPaths.cmake")
join_paths(palloc_pc_includedir "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")
join_paths(palloc_pc_libdir "\${prefix}" "${CMAKE_INSTALL_LIBDIR}")

configure_file(palloc.pc.in palloc.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/palloc.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")



# -----------------------------------------------------------------------------
# API surface testing
# -----------------------------------------------------------------------------

if (PALLOC_BUILD_TESTS)
  enable_testing()

  # static link tests
  foreach(TEST_NAME api api-fill stress)
    add_executable(palloc-test-${TEST_NAME} test/test-${TEST_NAME}.c)
    target_compile_definitions(palloc-test-${TEST_NAME} PRIVATE ${palloc_defines})
    target_compile_options(palloc-test-${TEST_NAME} PRIVATE ${palloc_cflags})
    target_include_directories(palloc-test-${TEST_NAME} PRIVATE include)
    if(PALLOC_BUILD_STATIC AND NOT PALLOC_DEBUG_TSAN)
      target_link_libraries(palloc-test-${TEST_NAME} PRIVATE palloc-static ${palloc_libraries})
    elseif(PALLOC_BUILD_SHARED)
      target_link_libraries(palloc-test-${TEST_NAME} PRIVATE palloc ${palloc_libraries})
    else()
      message(STATUS "cannot build TSAN tests without PALLOC_BUILD_SHARED being enabled")
    endif()
    add_test(NAME test-${TEST_NAME} COMMAND palloc-test-${TEST_NAME})
  endforeach()

  # dynamic override test
  if(PALLOC_BUILD_SHARED AND NOT (PALLOC_TRACK_ASAN OR PALLOC_DEBUG_TSAN OR PALLOC_DEBUG_UBSAN))
    add_executable(palloc-test-stress-dynamic test/test-stress.c)
    target_compile_definitions(palloc-test-stress-dynamic PRIVATE ${palloc_defines} "USE_STD_MALLOC=1")
    target_compile_options(palloc-test-stress-dynamic PRIVATE ${palloc_cflags})
    target_include_directories(palloc-test-stress-dynamic PRIVATE include)
    if(WIN32)
      target_compile_definitions(palloc-test-stress-dynamic PRIVATE "PALLOC_LINK_VERSION=1")  # link palloc_version
      target_link_libraries(palloc-test-stress-dynamic PRIVATE palloc ${palloc_libraries})  # link palloc_version
      add_test(NAME test-stress-dynamic COMMAND ${CMAKE_COMMAND} -E env PALLOC_VERBOSE=1 $<TARGET_FILE:palloc-test-stress-dynamic>)
    else()
      target_link_libraries(palloc-test-stress-dynamic PRIVATE ${palloc_libraries}) # pthreads, issue 1158
      if(APPLE)
        set(LD_PRELOAD "DYLD_INSERT_LIBRARIES")
      else()
        set(LD_PRELOAD "LD_PRELOAD")
      endif()
      add_test(NAME test-stress-dynamic COMMAND ${CMAKE_COMMAND} -E env PALLOC_VERBOSE=1 ${LD_PRELOAD}=$<TARGET_FILE:palloc> $<TARGET_FILE:palloc-test-stress-dynamic>)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# Set override properties
# -----------------------------------------------------------------------------
if (PALLOC_OVERRIDE)
  if (PALLOC_BUILD_SHARED)
    target_compile_definitions(palloc PRIVATE PALLOC_MALLOC_OVERRIDE)
  endif()
  if(NOT WIN32)
    # It is only possible to override malloc on Windows when building as a DLL.
    if (PALLOC_BUILD_STATIC)
      target_compile_definitions(palloc-static PRIVATE PALLOC_MALLOC_OVERRIDE)
    endif()
    if (PALLOC_BUILD_OBJECT)
      target_compile_definitions(palloc-obj PRIVATE PALLOC_MALLOC_OVERRIDE)
    endif()
  endif()
endif()
