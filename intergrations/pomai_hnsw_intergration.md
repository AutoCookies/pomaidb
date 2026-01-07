```cpp
// Example integration notes for PPHNSW (pseudo-patch)

// global / member
std::unique_ptr<pomai::ai::PPPQ> pp_pq;
size_t max_elems = 1'200'000;
pp_pq.reset(new pomai::ai::PPPQ(/*dim=*/512, /*m=*/8, /*k=*/256, max_elems, "pppq_codes.mmap"));

// On startup: train codebooks (example using random sample)
std::vector<float> samples(n_samples * dim);
// ... fill samples ...
pp_pq->train(samples.data(), n_samples);

// In addPoint
void addPoint(const float* vec, labeltype label) {
  // label used as id
  pp_pq->addVec(vec, label);
  // Add HNSW internal structure as normal (you may omit storing full vector in RAM
  // if you will always use approxDist during search).
}

// In distance function used by HNSW search:
float pq_distance(labeltype a, labeltype b) {
  return pp_pq->approxDist(a, b);
}

// Periodically (e.g., background thread) call:
// pp_pq->purgeCold(1000 /*ms threshold*/);