# In-place fix for third_party/palloc/src/init.c: designator order must match pa_page_s
# when built with C++ (PA_USE_CXX=ON). Replaces the _pa_page_empty initializer block.
set(INIT_FILE "${PALLOC_SRC_DIR}/src/init.c")
file(READ "${INIT_FILE}" CONTENT)
# Already patched?
if (CONTENT MATCHES "\\.heap_tag = 0")
  return()
endif()
# Block to find and replace (vanilla order)
set(OLD_BLOCK "// Empty page used to initialize the small free pages array
const pa_page_t _pa_page_empty = {
  .slice_count = 0,
  .slice_offset = 0,
  .is_committed = false,
  .is_huge = false,
  .is_zero_init = false,
  .retire_expire = 0,
  .capacity = 0,
  .used = 0,
  .reserved = 0,
  .flags = { 0 },
  .free_is_zero = false,
  .block_size_shift = 0,
  .free = NULL,
  .local_free = NULL,
  .block_size = 0,
  .page_start = NULL,
  #if (PA_PADDING || PA_ENCODE_FREELIST)
  .keys = { 0, 0 },
  #endif
  .xthread_free = PA_ATOMIC_VAR_INIT(0),
  .xheap = PA_ATOMIC_VAR_INIT(0),
  .next = NULL,
  .prev = NULL,
  .padding = { 0 }
};")
set(NEW_BLOCK "// Empty page used to initialize the small free pages array (designator order matches pa_page_s in types.h)
const pa_page_t _pa_page_empty = {
  .slice_count = 0,
  .slice_offset = 0,
  .is_committed = false,
  .is_huge = false,
  .is_zero_init = false,
  .retire_expire = 0,
  .capacity = 0,
  .used = 0,
  .flags = { 0 },
  .free_is_zero = false,
  .block_size_shift = 0,
  .free = NULL,
  .local_free = NULL,
  .xthread_free = PA_ATOMIC_VAR_INIT(0),
  .block_size = 0,
  #if (PA_PADDING || PA_ENCODE_FREELIST)
  .keys = { 0, 0 },
  #endif
  .reserved = 0,
  .heap_tag = 0,
  .page_start = NULL,
  .xheap = PA_ATOMIC_VAR_INIT(0),
  .next = NULL,
  .prev = NULL,
  .padding = { 0 }
};")
string(REPLACE "${OLD_BLOCK}" "${NEW_BLOCK}" NEW_CONTENT "${CONTENT}")
if (NEW_CONTENT STREQUAL "${CONTENT}")
  message(FATAL_ERROR "palloc init.c: could not find expected block to patch (designator order fix)")
endif()
file(WRITE "${INIT_FILE}" "${NEW_CONTENT}")
message(STATUS "[pomai] palloc: applied init.c designator-order fix")
