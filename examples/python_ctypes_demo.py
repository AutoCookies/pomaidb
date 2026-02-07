import ctypes

lib = ctypes.CDLL("./libpomai_c.so")

print("abi:", lib.pomai_abi_version())
lib.pomai_version_string.restype = ctypes.c_char_p
print("version:", lib.pomai_version_string().decode())
