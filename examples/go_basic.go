// PomaiDB Go cgo example.
//
// How to run:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target pomai_c
//   export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
//   go run examples/go_basic.go
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../build -lpomai_c
#include "pomai/c_api.h"
*/
import "C"

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"unsafe"
)

func checkStatus(st *C.pomai_status_t) {
	if st == nil {
		return
	}
	msg := C.pomai_status_message(st)
	defer C.pomai_status_free(st)
	panic(C.GoString(msg))
}

func main() {
	var opts C.pomai_options_t
	C.pomai_options_init(&opts)
	opts.struct_size = C.uint32_t(unsafe.Sizeof(opts))
	opts.path = C.CString(filepath.Clean("/tmp/pomai_example_go"))
	opts.shards = 4
	opts.dim = 8
	opts.search_threads = 2

	var db *C.pomai_db_t
	checkStatus(C.pomai_open(&opts, &db))

	rand.Seed(2024)
	dim := int(opts.dim)
	n := 100
	vectors := make([][]float32, n)
	upserts := make([]C.pomai_upsert_t, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1
		}
		vectors[i] = vec
		upserts[i].struct_size = C.uint32_t(unsafe.Sizeof(upserts[i]))
		upserts[i].id = C.uint64_t(i)
		upserts[i].vector = (*C.float)(unsafe.Pointer(&vec[0]))
		upserts[i].dim = C.uint32_t(dim)
		upserts[i].metadata = nil
		upserts[i].metadata_len = 0
	}
	checkStatus(C.pomai_put_batch(db, &upserts[0], C.size_t(n)))
	checkStatus(C.pomai_freeze(db))

	query := C.pomai_query_t{}
	query.struct_size = C.uint32_t(unsafe.Sizeof(query))
	query.vector = (*C.float)(unsafe.Pointer(&vectors[0][0]))
	query.dim = C.uint32_t(dim)
	query.topk = 5
	query.filter_expression = nil
	query.alpha = 1.0
	query.deadline_ms = 0

	var results *C.pomai_search_results_t
	checkStatus(C.pomai_search(db, &query, &results))

	count := int(results.count)
	ids := unsafe.Slice(results.ids, count)
	scores := unsafe.Slice(results.scores, count)
	fmt.Println("TopK results:")
	for i := 0; i < count; i++ {
		fmt.Printf("  id=%d score=%.4f\n", uint64(ids[i]), float32(scores[i]))
	}
	C.pomai_search_results_free(results)
	checkStatus(C.pomai_close(db))
}
