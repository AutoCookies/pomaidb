// PomaiDB TypeScript FFI example (Node.js + ts-node).
//
// How to run:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target pomai_c
//   npm install ffi-napi ref-napi ref-struct-di ts-node typescript
//   POMAI_C_LIB=./build/libpomai_c.so npx ts-node --compiler-options '{"module":"commonjs"}' examples/ts_basic.ts

import ffi from "ffi-napi";
import ref from "ref-napi";
import StructDi from "ref-struct-di";
import path from "node:path";

const Struct = StructDi(ref);
const voidPtr = ref.refType(ref.types.void);
const floatPtr = ref.refType(ref.types.float);
const uint8Ptr = ref.refType(ref.types.uint8);

const PomaiOptions = Struct({
  struct_size: ref.types.uint32,
  path: ref.types.CString,
  shards: ref.types.uint32,
  dim: ref.types.uint32,
  search_threads: ref.types.uint32,
  fsync_policy: ref.types.uint32,
  memory_budget_bytes: ref.types.uint64,
  deadline_ms: ref.types.uint32,
});

const PomaiUpsert = Struct({
  struct_size: ref.types.uint32,
  id: ref.types.uint64,
  vector: floatPtr,
  dim: ref.types.uint32,
  metadata: uint8Ptr,
  metadata_len: ref.types.uint32,
});

const PomaiQuery = Struct({
  struct_size: ref.types.uint32,
  vector: floatPtr,
  dim: ref.types.uint32,
  topk: ref.types.uint32,
  filter_expression: ref.types.CString,
  alpha: ref.types.float,
  deadline_ms: ref.types.uint32,
});

const PomaiSearchResults = Struct({
  struct_size: ref.types.uint32,
  count: ref.types.size_t,
  ids: ref.refType(ref.types.uint64),
  scores: floatPtr,
  shard_ids: ref.refType(ref.types.uint32),
});

const libPath = process.env.POMAI_C_LIB ?? path.resolve("./build/libpomai_c.so");
const lib = ffi.Library(libPath, {
  pomai_options_init: ["void", [ref.refType(PomaiOptions)]],
  pomai_open: [voidPtr, [ref.refType(PomaiOptions), ref.refType(voidPtr)]],
  pomai_close: [voidPtr, [voidPtr]],
  pomai_put: [voidPtr, [voidPtr, ref.refType(PomaiUpsert)]],
  pomai_freeze: [voidPtr, [voidPtr]],
  pomai_search: [voidPtr, [voidPtr, ref.refType(PomaiQuery), ref.refType(ref.refType(PomaiSearchResults))]],
  pomai_search_results_free: ["void", [ref.refType(PomaiSearchResults)]],
  pomai_status_message: ["string", [voidPtr]],
  pomai_status_free: ["void", [voidPtr]],
});

function checkStatus(status: any) {
  if (!ref.isNull(status)) {
    const msg = lib.pomai_status_message(status);
    lib.pomai_status_free(status);
    throw new Error(msg);
  }
}

function makeVector(dim: number, seed: number) {
  const buf = Buffer.alloc(dim * 4);
  const view = new Float32Array(buf.buffer, buf.byteOffset, dim);
  let next = seed;
  for (let i = 0; i < dim; i += 1) {
    next = (next * 1664525 + 1013904223) >>> 0;
    view[i] = ((next % 1000) / 500) - 1;
  }
  return { buf, seed: next };
}

const dim = 8;
const total = 50;

const opts = new PomaiOptions();
lib.pomai_options_init(opts.ref());
opts.struct_size = PomaiOptions.size;
opts.path = path.resolve("/tmp/pomai_example_ts");
opts.shards = 4;
opts.dim = dim;
opts.search_threads = 2;

const dbPtr = ref.alloc(voidPtr);
checkStatus(lib.pomai_open(opts.ref(), dbPtr));
const db = dbPtr.deref();

let seed = 4242;
const vectors: Buffer[] = [];
for (let i = 0; i < total; i += 1) {
  const out = makeVector(dim, seed);
  seed = out.seed;
  vectors.push(out.buf);
  const upsert = new PomaiUpsert();
  upsert.struct_size = PomaiUpsert.size;
  upsert.id = i;
  upsert.vector = out.buf;
  upsert.dim = dim;
  upsert.metadata = ref.NULL;
  upsert.metadata_len = 0;
  checkStatus(lib.pomai_put(db, upsert.ref()));
}
checkStatus(lib.pomai_freeze(db));

const query = new PomaiQuery();
query.struct_size = PomaiQuery.size;
query.vector = vectors[0];
query.dim = dim;
query.topk = 5;
query.filter_expression = ref.NULL;
query.alpha = 1.0;
query.deadline_ms = 0;

const resultsPtrPtr = ref.alloc(ref.refType(PomaiSearchResults));
checkStatus(lib.pomai_search(db, query.ref(), resultsPtrPtr));
const resultsPtr = resultsPtrPtr.deref();
const results = resultsPtr.deref();

const idsBuf = ref.reinterpret(results.ids, Number(results.count) * 8, 0);
const scoresBuf = ref.reinterpret(results.scores, Number(results.count) * 4, 0);
const ids = new BigUint64Array(idsBuf.buffer, idsBuf.byteOffset, Number(results.count));
const scores = new Float32Array(scoresBuf.buffer, scoresBuf.byteOffset, Number(results.count));

console.log("TopK results:");
for (let i = 0; i < results.count; i += 1) {
  console.log(`  id=${ids[i].toString()} score=${scores[i].toFixed(4)}`);
}

lib.pomai_search_results_free(resultsPtr);
checkStatus(lib.pomai_close(db));
