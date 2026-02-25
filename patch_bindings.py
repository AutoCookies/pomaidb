import os
import re

py_query_pattern = r'class PomaiQuery\(ctypes\.Structure\):\n\s*_fields_\s*=\s*\[\n(\s*\("struct_size",[^\]]+?\n(\s*)\("deadline_ms"[^\n]+)\n\s*\]'
py_query_repl = r'class PomaiQuery(ctypes.Structure):\n    _fields_ = [\n\1,\n\2("flags", ctypes.c_uint32),\n    ]'

py_results_pattern = r'class PomaiSearchResults\(ctypes\.Structure\):\n\s*_fields_\s*=\s*\[\n([\s\S]+?\("shard_ids"[^\n]+)\n\s*\]'
py_results_repl = r'''class PomaiSemanticPointer(ctypes.Structure):
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
\1,
        ("zero_copy_pointers", ctypes.POINTER(PomaiSemanticPointer)),
    ]'''

js_query_pattern = r'const PomaiQuery = Struct\(\{([\s\S]+?deadline_ms:\s*ref\.types\.uint32),?\n\}\);'
js_query_repl = r'const PomaiQuery = Struct({\1,\n  flags: ref.types.uint32,\n});'

js_results_pattern = r'const PomaiSearchResults = Struct\(\{([\s\S]+?shard_ids:[^\n]+)\n\}\);'
js_results_repl = r'''const PomaiSemanticPointer = Struct({
  struct_size: ref.types.uint32,
  raw_data_ptr: voidPtr,
  dim: ref.types.uint32,
  quant_min: ref.types.float,
  quant_inv_scale: ref.types.float,
  session_id: ref.types.uint64,
});

const PomaiSearchResults = Struct({\1,
  zero_copy_pointers: ref.refType(PomaiSemanticPointer),
});'''

def patch_file(p):
    with open(p, 'r') as f:
        content = f.read()
    
    orig = content
    if p.endswith('.py') and "PomaiQuery" in content:
        content = re.sub(py_query_pattern, py_query_repl, content)
        if "PomaiSemanticPointer" not in content:
            content = re.sub(py_results_pattern, py_results_repl, content)
            
    if (p.endswith('.ts') or p.endswith('.mjs')) and "PomaiQuery" in content:
        content = re.sub(js_query_pattern, js_query_repl, content)
        if "PomaiSemanticPointer" not in content:
            content = re.sub(js_results_pattern, js_results_repl, content)
            
    if orig != content:
        with open(p, 'w') as f:
            f.write(content)
        print(f"Patched {p}")

for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py') or f.endswith('.ts') or f.endswith('.mjs'):
            patch_file(os.path.join(root, f))
