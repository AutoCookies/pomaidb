import pomaidb
import tempfile
import time

dirpath = tempfile.mkdtemp()
mem = pomaidb.agent_memory_open(dirpath + "/am", dim=128)
print("Opened.")
pomaidb.agent_memory_append(mem, "agent_1", "sess_1", "msg", 1, "hello", [0.1]*128)
print("Appended.")
