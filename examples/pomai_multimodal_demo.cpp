#include "pomai/pomai.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Starting PomaiDB Multi-modal Demo..." << std::endl;

    pomai::DBOptions opt;
    opt.path = "./multimodal_db";
    opt.dim = 4;
    opt.shard_count = 1;
    opt.index_params.quant_type = pomai::QuantizationType::kBit;

    std::unique_ptr<pomai::DB> db;
    pomai::Status st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Open failed: " << st.ToString() << std::endl;
        return 1;
    }

    // 1. Ingest Data with Temporal, Lexical, Document, and Spatial metadata
    // Vector 1: "The cat sat on the mat" @ 10:00 AM, Location: Paris
    {
        std::vector<float> v1 = {0.1f, 0.2f, 0.3f, 0.4f};
        pomai::Metadata m1;
        m1.timestamp = 1000;
        m1.text = "The cat sat on the mat";
        m1.payload = "{\"id\": \"doc_001\", \"type\": \"observation\"}";
        m1.lat = 48.8566;
        m1.lon = 2.3522;
        (void)db->Put(1, v1, m1);
    }

    // Vector 2: "A dog barks in the garden" @ 11:00 AM, Location: London
    {
        std::vector<float> v2 = {0.5f, 0.6f, 0.7f, 0.8f};
        pomai::Metadata m2;
        m2.timestamp = 2000;
        m2.text = "A dog barks in the garden";
        m2.payload = "{\"id\": \"doc_002\", \"type\": \"alert\"}";
        m2.lat = 51.5074;
        m2.lon = -0.1278;
        (void)db->Put(2, v2, m2);
    }

    st = db->Freeze("__default__");
    if (!st.ok()) {
        std::cerr << "Freeze failed: " << st.message() << std::endl;
        return 1;
    }

    // 2. Hybrid Search (Vector + Lexical)
    std::cout << "\nTesting Hybrid Search (Query: 'garden')..." << std::endl;
    pomai::MultiModalQuery q1;
    q1.vector = {0.5f, 0.6f, 0.7f, 0.8f}; // Matches doc_2 semantically
    q1.keywords = "garden";               // Matches doc_2 lexically
    q1.alpha = 0.5f;                      // Equal weight
    
    pomai::SearchResult res1;
    st = db->SearchMultiModal(q1, &res1);
    std::cout << "Results: " << res1.hits.size() << " hits" << std::endl;
    for (const auto& hit : res1.hits) {
        std::cout << " - ID: " << hit.id << ", Score: " << hit.score << std::endl;
    }

    // 3. Temporal Filtering
    std::cout << "\nTesting Temporal Filter (Time: [1500, 2500])..." << std::endl;
    pomai::MultiModalQuery q2;
    q2.vector = {0.1f, 0.2f, 0.3f, 0.4f};
    q2.start_ts = 1500;
    q2.end_ts = 2500;
    
    pomai::SearchResult res2;
    st = db->SearchMultiModal(q2, &res2);
    std::cout << "Results in range: " << res2.hits.size() << " (Expected: 1, doc_2)" << std::endl;
    for (const auto& hit : res2.hits) {
        std::cout << " - ID: " << hit.id << std::endl;
    }

    // 4. Spatial Filtering (Radius)
    std::cout << "\nTesting Spatial Filter (Radius: 100km from Paris)..." << std::endl;
    // We can't pass spatial filter directly via MultiModalQuery yet, 
    // but we can use SearchVector with Filter::Radius.
    pomai::SearchOptions sopt;
    sopt.filters.push_back(pomai::Filter::Radius(48.8566, 2.3522, 100.0));
    
    std::vector<float> q3 = {0.1f, 0.2f, 0.3f, 0.4f};
    pomai::SearchResult res3;
    st = db->Search(std::span<const float>(q3), 10, sopt, &res3);
    std::cout << "Results in radius: " << res3.hits.size() << " (Expected: 1, doc_1)" << std::endl;
    for (const auto& hit : res3.hits) {
        std::cout << " - ID: " << hit.id << std::endl;
    }

    st = db->Close();
    if (!st.ok()) {
        std::cerr << "Close failed: " << st.message() << std::endl;
        return 1;
    }
    std::cout << "\nMulti-modal Verification Complete!" << std::endl;
    return 0;
}
