#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "pomai/pomai.h"
#include "util/posix_file.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 100) return 0;

  std::string fuzz_dir = "/tmp/pomai_fuzz_storage_" + std::to_string(getpid());
  system(("mkdir -p " + fuzz_dir).c_str());

  // Use fuzzed data to create a "corrupted" segment or manifest
  std::string seg_path = fuzz_dir + "/corrupted.seg";
  std::ofstream seg_file(seg_path, std::ios::binary);
  seg_file.write(reinterpret_cast<const char*>(data), size);
  seg_file.close();

  // Also create a "corrupted" manifest that points to it
  std::string manifest_path = fuzz_dir + "/MANIFEST";
  std::ofstream manifest_file(manifest_path);
  manifest_file << "corrupted.seg" << std::endl;
  manifest_file.close();

  // Try to open a DB in this directory
  pomai::DBOptions opts;
  opts.path = fuzz_dir;
  opts.dim = 128; // fixed for this fuzzer
  opts.shard_count = 1;

  std::unique_ptr<pomai::DB> db;
  // This might fail with Status error, which is GOOD. It should NOT segfault.
  auto st = pomai::DB::Open(opts, &db);
  
  if (st.ok() && db) {
      // If it opened, try a search or freeze to force reading the segment
      std::vector<float> query(128, 0.0f);
      pomai::SearchResult res;
      db->Search(query, 5, &res);
      db->Freeze("default");
      db->Close();
  }

  // Cleanup
  system(("rm -rf " + fuzz_dir).c_str());

  return 0;
}
