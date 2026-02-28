#!/usr/bin/env python3
"""
PomaiDB RAG quick start: create a RAG membrane, add chunks (text→token IDs + optional embeddings), search.
Run from repo root after building: POMAI_C_LIB=build/libpomai_c.so python3 examples/rag_quickstart.py
"""
import sys
import tempfile
from pathlib import Path

# Prefer repo python package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import pomaidb

def main():
    with tempfile.TemporaryDirectory(prefix="pomai_rag_") as tmp:
        db = pomaidb.open_db(tmp + "/db", dim=4, shards=1)

        # Create a RAG membrane (name, embedding dim, shard count)
        pomaidb.create_rag_membrane(db, "docs", dim=4, shard_count=1)

        # Chunk 1: doc_id=1, chunk_id=10, tokens [100, 200], optional embedding
        pomaidb.put_chunk(db, "docs", chunk_id=10, doc_id=1, token_ids=[100, 200], vector=[1.0, 0.0, 0.0, 0.0])

        # Chunk 2: doc_id=2, chunk_id=20, tokens [200, 300]
        pomaidb.put_chunk(db, "docs", chunk_id=20, doc_id=2, token_ids=[200, 300], vector=[0.0, 1.0, 0.0, 0.0])

        # Search by token overlap: query tokens [200] → matches both chunks; with vector rerank
        hits = pomaidb.search_rag(db, "docs", token_ids=[200], vector=[0.5, 0.5, 0.0, 0.0], topk=5)
        print("RAG search (token 200 + vector):", hits)

        pomaidb.close(db)
    print("OK")

if __name__ == "__main__":
    main()
