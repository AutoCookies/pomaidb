#!/usr/bin/env python3
"""RAG smoke test for CI: create RAG membrane, put chunks, search_rag. Fails on error."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

def main():
    lib = os.environ.get("POMAI_C_LIB", str(ROOT / "build" / "libpomai_c.so"))
    if not os.path.isfile(lib):
        print("SKIP: lib not found:", lib)
        return 0

    import pomaidb

    with tempfile.TemporaryDirectory(prefix="pomai_rag_smoke_") as tmp:
        db = pomaidb.open_db(tmp + "/db", dim=4, shards=1)
        pomaidb.create_rag_membrane(db, "rag", dim=4, shard_count=1)
        pomaidb.put_chunk(db, "rag", chunk_id=1, doc_id=1, token_ids=[10, 20], vector=[1.0, 0.0, 0.0, 0.0])
        pomaidb.put_chunk(db, "rag", chunk_id=2, doc_id=1, token_ids=[20, 30], vector=[0.0, 1.0, 0.0, 0.0])
        hits = pomaidb.search_rag(db, "rag", token_ids=[20], vector=[0.5, 0.5, 0.0, 0.0], topk=5)
        pomaidb.close(db)

    if len(hits) < 1:
        print("FAIL: expected at least 1 hit, got", hits)
        return 1
    print("RAG smoke OK:", len(hits), "hits")
    return 0

if __name__ == "__main__":
    sys.exit(main())
