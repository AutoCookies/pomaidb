import ctypes
import numpy as np
import time
import os

# Define structs
class PomaiQuery(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("dim", ctypes.c_uint32),
        ("topk", ctypes.c_uint32),
        ("filter_expression", ctypes.c_char_p),
        ("alpha", ctypes.c_float),
        ("deadline_ms", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]

class PomaiOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("path", ctypes.c_char_p),
        ("shards", ctypes.c_uint32),
        ("dim", ctypes.c_uint32),
        ("search_threads", ctypes.c_uint32),
        ("fsync_policy", ctypes.c_int),
        ("memory_budget_bytes", ctypes.c_uint64),
        ("deadline_ms", ctypes.c_uint32),
    ]

class PomaiUpsert(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("dim", ctypes.c_uint32),
        ("metadata", ctypes.POINTER(ctypes.c_uint8)),
        ("metadata_len", ctypes.c_uint32),
    ]

class PomaiSemanticPointer(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("raw_data_ptr", ctypes.c_void_p),
        ("dim", ctypes.c_uint32),
        ("quant_min", ctypes.c_float),
        ("quant_inv_scale", ctypes.c_float),
        ("session_id", ctypes.c_uint64),
    ]

class PomaiSearchResults(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("count", ctypes.c_size_t),
        ("ids", ctypes.POINTER(ctypes.c_uint64)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("shard_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("zero_copy_pointers", ctypes.POINTER(PomaiSemanticPointer)),
    ]

# Load library
lib = ctypes.CDLL("./build/libpomai_c.so")
lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]

lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
lib.pomai_open.restype = ctypes.c_int

lib.pomai_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert)]
lib.pomai_put.restype = ctypes.c_void_p

lib.pomai_freeze.argtypes = [ctypes.c_void_p]
lib.pomai_freeze.restype = ctypes.c_void_p

lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
lib.pomai_search.restype = ctypes.c_void_p

lib.pomai_status_message.argtypes = [ctypes.c_void_p]
lib.pomai_status_message.restype = ctypes.c_char_p

lib.pomai_status_free.argtypes = [ctypes.c_void_p]

lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
lib.pomai_release_pointer.argtypes = [ctypes.c_uint64]

lib.pomai_close.argtypes = [ctypes.c_void_p]

def main():
    db_path = b"test_db_zero_copy"
    print("Opening PomaiDB instance...")
    
    opts = PomaiOptions()
    lib.pomai_options_init(ctypes.byref(opts))
    opts.path = db_path
    opts.shards = 1
    opts.dim = 128
    
    db = ctypes.c_void_p()
    if lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)) != 0:
        print("Failed to open db")
        return
    
    dim = 128
    
    print("Inserting vector...")
    vec = (ctypes.c_float * dim)(*[float(x*0.01) for x in range(dim)])
    
    upsert = PomaiUpsert()
    upsert.struct_size = ctypes.sizeof(PomaiUpsert)
    upsert.id = 42
    upsert.vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
    upsert.dim = dim
    upsert.metadata = None
    upsert.metadata_len = 0
    lib.pomai_put(db, ctypes.byref(upsert))
    
    print("Freezing DB to guarantee vector is bundled into a Segment...")
    freeze_st = lib.pomai_freeze(db)
    if freeze_st:
        print("Pomai freeze failed")
        lib.pomai_status_free(freeze_st)
        return
    
    print("Performing Zero-Copy Search...")
    q = PomaiQuery()
    q.struct_size = ctypes.sizeof(PomaiQuery)
    q.vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
    q.dim = dim
    q.topk = 5
    q.filter_expression = None
    q.alpha = 0.0
    q.deadline_ms = 0
    q.topk = 5
    POMAI_QUERY_FLAG_ZERO_COPY = 1
    q.flags = POMAI_QUERY_FLAG_ZERO_COPY
    
    res = ctypes.POINTER(PomaiSearchResults)()
    status = lib.pomai_search(db, ctypes.byref(q), ctypes.byref(res))
    
    if not status and res:
        count = res.contents.count
        print(f"Search successful. Found {count} hits.")
        if count > 0:
            ptr_struct = res.contents.zero_copy_pointers[0]
            if ptr_struct.raw_data_ptr:
                print(f"Got SemanticPointer for Hit 0! Memory session ID: {ptr_struct.session_id}")
                
                # Numpy from_address to demonstrate zero copy
                ArrayType = ctypes.c_uint8 * dim  # SQ8
                buffer = ArrayType.from_address(ptr_struct.raw_data_ptr)
                np_array = np.frombuffer(buffer, dtype=np.uint8)
                
                print(f"Decoded SQ8 array values (head): {np_array[:5]}")
                print(f"Quantization Params: min={ptr_struct.quant_min}, inv_scale={ptr_struct.quant_inv_scale}")
                
                # Dequantize using numpy
                dequantized = np_array * ptr_struct.quant_inv_scale + ptr_struct.quant_min
                print(f"Dequantized floats (head): {dequantized[:5]}")
                print(f"Original floats (head): {[vec[x] for x in range(5)]}")
                
                # Release pointer
                print("Releasing zero-copy pointer memory...")
                lib.pomai_release_pointer(ptr_struct.session_id)
            else:
                print("SemanticPointer raw_data_ptr is NULL (likely hit is inside MemTable and not yet in Segment)")
        lib.pomai_search_results_free(res)
    else:
        err_msg = lib.pomai_status_message(status).decode()
        print("Search failed with status message:", err_msg)
        lib.pomai_status_free(status)
        
    lib.pomai_close(db)

if __name__ == "__main__":
    main()
