#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>

#include "pomai/pomai.h"

// Fuzzer random generator helper (inspired by RocksDB)
class FuzzedDataProvider {
public:
  FuzzedDataProvider(const uint8_t* data, size_t size) : data_(data), size_(size), offset_(0) {}

  template <typename T>
  T ConsumeIntegralInRange(T min, T max) {
    if (offset_ + sizeof(T) > size_) return min;
    T val;
    memcpy(&val, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return min + (val % (max - min + 1));
  }

  std::vector<float> ConsumeFloats(size_t n) {
    std::vector<float> res;
    res.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (offset_ + 4 > size_) {
        res.push_back(0.0f);
        continue;
      }
      float f;
      memcpy(&f, data_ + offset_, 4);
      offset_ += 4;
      
      // Occasionally inject NaN/Inf
      uint32_t roll = ConsumeIntegralInRange<uint32_t>(0, 100);
      if (roll == 0) f = NAN;
      else if (roll == 1) f = INFINITY;
      else if (roll == 2) f = -INFINITY;
      
      res.push_back(f);
    }
    return res;
  }

  std::string ConsumeRandomString(size_t max_len) {
    size_t len = ConsumeIntegralInRange<size_t>(0, max_len);
    if (offset_ + len > size_) len = size_ - offset_;
    std::string s(reinterpret_cast<const char*>(data_ + offset_), len);
    offset_ += len;
    return s;
  }

  bool ConsumeBool() {
    return ConsumeIntegralInRange<uint8_t>(0, 1) == 1;
  }

  uint8_t ConsumeByte() {
    if (offset_ >= size_) return 0;
    return data_[offset_++];
  }

  size_t RemainingBytes() { return size_ - offset_; }

private:
  const uint8_t* data_;
  size_t size_;
  size_t offset_;
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 16) return 0;

  FuzzedDataProvider fuzzed(data, size);
  
  // Setup DB
  std::string db_path = "/tmp/pomai_fuzz_membrane_" + std::to_string(getpid());
  pomai::DBOptions opts;
  opts.path = db_path;
  opts.dim = fuzzed.ConsumeIntegralInRange<uint32_t>(1, 1024);
  opts.shard_count = fuzzed.ConsumeIntegralInRange<uint32_t>(1, 8);
  
  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opts, &db);
  if (!st.ok()) return 0;

  // Interleaved operations
  while (fuzzed.RemainingBytes() > opts.dim * 4 + 16) {
    uint8_t op = fuzzed.ConsumeByte() % 4;
    pomai::VectorId id = fuzzed.ConsumeIntegralInRange<pomai::VectorId>(0, 100000);
    
    if (op == 0) { // Put
      auto vec = fuzzed.ConsumeFloats(opts.dim);
      pomai::Metadata meta;
      meta.tenant = fuzzed.ConsumeRandomString(128);
      db->Put(id, vec, meta);
    } 
    else if (op == 1) { // Search
      auto query = fuzzed.ConsumeFloats(opts.dim);
      pomai::SearchResult res;
      db->Search(query, 10, &res);
    }
    else if (op == 2) { // Delete
      db->Delete(id);
    }
    else if (op == 3) { // Freeze
      db->Freeze("default");
    }
  }

  db->Close();
  // Cleanup
  std::string rm_cmd = "rm -rf " + db_path;
  system(rm_cmd.c_str());

  return 0;
}
