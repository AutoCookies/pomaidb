#!/usr/bin/env python3
"""
PomaiDB RAG quick start: offline ingest_document + retrieve_context (no external API).
Optionally use put_chunk + search_rag for low-level control.
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
        db = pomaidb.open_db(tmp + "/db", dim=4, shards=2)

        # Create a RAG membrane (name, embedding dim, shard count)
        pomaidb.create_rag_membrane(db, "docs", dim=4, shard_count=2)

        # Offline ingest: chunk + embed (mock) + tokenize + store
        pomaidb.ingest_document(db, "docs", doc_id=1, text="Alpha content here. Chunking and embedding are automatic.")
        pomaidb.ingest_document(db, "docs", doc_id=2, text="Beta content there. Retrieve context returns stored chunk text.")

        # Retrieve context for a query (embed query, search, format chunk text)
        context = pomaidb.retrieve_context(db, "docs", "content", top_k=5)
        print("Retrieve context for 'content':", repr(context[:200]) + ("..." if len(context) > 200 else ""))

        # Low-level: put_chunk + search_rag (optional)
        pomaidb.put_chunk(db, "docs", chunk_id=100, doc_id=3, token_ids=[100, 200], vector=[1.0, 0.0, 0.0, 0.0])
        hits = pomaidb.search_rag(db, "docs", token_ids=[200], vector=[0.5, 0.5, 0.0, 0.0], topk=5)
        print("RAG search (token 200 + vector):", [(h[0], h[1], h[2], h[3]) for h in hits])

        pomaidb.close(db)
    print("OK")

if __name__ == "__main__":
    main()
