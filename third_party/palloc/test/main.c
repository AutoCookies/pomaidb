#include <stdio.h>
#include <assert.h>
#include <palloc.h>

void test_heap(void* p_out) {
  palloc_heap_t* heap = palloc_heap_new();
  void* p1 = palloc_heap_malloc(heap,32);
  void* p2 = palloc_heap_malloc(heap,48);
  palloc_free(p_out);
  palloc_heap_destroy(heap);
  //palloc_heap_delete(heap); palloc_free(p1); palloc_free(p2);
}

void test_large() {
  const size_t N = 1000;

  for (size_t i = 0; i < N; ++i) {
    size_t sz = 1ull << 21;
    char* a = palloc_mallocn_tp(char,sz);
    for (size_t k = 0; k < sz; k++) { a[k] = 'x'; }
    palloc_free(a);
  }
}

int main() {
  void* p1 = palloc_malloc(16);
  void* p2 = palloc_malloc(1000000);
  palloc_free(p1);
  palloc_free(p2);
  p1 = palloc_malloc(16);
  p2 = palloc_malloc(16);
  palloc_free(p1);
  palloc_free(p2);

  test_heap(palloc_malloc(32));

  p1 = palloc_malloc_aligned(64, 16);
  p2 = palloc_malloc_aligned(160,24);
  palloc_free(p2);
  palloc_free(p1);
  //test_large();

  palloc_collect(true);
  palloc_stats_print(NULL);
  return 0;
}
