Concrete list of required fixes to be fully correct / production-ready

    Fix packed4 truncation:
        In VectorStore::upsert, choose between raw 8-bit codes and packed4 depending on pq_->k():
            If pq_->k() > 16: write raw codes to pq_codes block (SoA supports that).
            Else: write packed4 into pq_packed4 block.
        Also adjust SoA append_vector/validation to accept both forms (you already updated so it does).

    Persist codebooks and load on startup:
        Store the codebooks filename or embed codebooks in SoA.user_meta or codebooks block:
            Option A: write path into SoaMmapHeader.user_meta (you have user_meta field) when creating SoA or when saving codebooks; on open, read header.user_meta and call pq_->load_codebooks(path).
            Option B (preferred for portability): store the codebooks floats in the codebooks block inside the SoA file (SoaMmapHeader has codebooks_offset/size reserved). This makes the SoA file self-contained.
        Implement VectorStore::attach_soa / init to check for codebooks in SoA header and load into pq_ if present.

    If you want offline training usage:
        Provide an API or config to tell VectorStore to load codebooks from a given file (instead of retraining): e.g., VectorStore::init(..., const std::string &pq_codebooks_path) or read from config.
        Make sure PQ training is optional if codebooks file exists; don't retrain at runtime when codebooks file is present.

    Add tests:
        A unit test that trains a PQ with k>16 and verifies that raw 8-bit codes roundtrip via pq_codes block (append -> pq_codes_ptr -> unpack/compare).
        Prefilter integration test (you asked earlier): insert clustered vectors, ensure prefilter+PQ reduces scan size by 10–100x and recall > 95% for top-100. Use both k <= 16 and k > 16 cases.
