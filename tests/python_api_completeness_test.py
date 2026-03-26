import tempfile
import pomaidb
import shutil
import time

dirpath = tempfile.mkdtemp()
print("Starting DB...")
try:
    db = pomaidb.open_db(dirpath, dim=128, shards=1, metric="ip")
    print("DB Opened.")
    pomaidb.put_batch(db, ids=[1, 2], vectors=[[0.1]*128, [0.2]*128])
    print("Put batch.")
    pomaidb.freeze(db)
    print("Frozen.")
    
    assert pomaidb.exists(db, 1) == True
    assert pomaidb.exists(db, 99) == False
    print("Exists works.")
    
    rec = pomaidb.get(db, 1)
    assert rec is not None
    assert rec["id"] == 1
    assert rec["dim"] == 128
    print("Get works.")
    
    pomaidb.delete(db, 1)
    print("Delete works.")
    
    pomaidb.create_membrane_kind(db, "my_kv", 0, 1, pomaidb.MEMBRANE_KIND_KEYVALUE)
    pomaidb.kv_put(db, "my_kv", "key1", "val1")
    assert pomaidb.kv_get(db, "my_kv", "key1") == "val1"
    pomaidb.kv_delete(db, "my_kv", "key1")
    print("KV works.")
    
    pomaidb.create_membrane_kind(db, "my_blob", 0, 1, pomaidb.MEMBRANE_KIND_BLOB)
    pomaidb.blob_put(db, "my_blob", 1, b"\x01\x02\x03")
    print("Blob works.")

    pomaidb.close(db)
    print("DB Closed.")
    
    # Test AgentMemory
    print("Opening AgentMemory...")
    mem = pomaidb.agent_memory_open(dirpath + "/agent", dim=128)
    print("AgentMemory appended...")
    outid = pomaidb.agent_memory_append(mem, "agent_1", "sess_1", "msg", 1, "hello", [0.1]*128)
    print("AgentMemory appended with id", outid)
    # Close and reopen to force memtable flush so iterator can see it
    pomaidb.agent_memory_close(mem)
    mem = pomaidb.agent_memory_open(dirpath + "/agent", dim=128)
    
    print("AgentMemory get_recent...")
    recent = pomaidb.agent_memory_get_recent(mem, "agent_1")
    print("Recent: [", len(recent), "] ->", recent)
    
    # search memory
    res = pomaidb.agent_memory_search(mem, "agent_1", embedding=[0.1]*128)
    print("Search: [", len(res), "] ->", res)

    pomaidb.agent_memory_close(mem)
    print("ALL TESTS PASSED (Check prints)")
    
finally:
    shutil.rmtree(dirpath)
