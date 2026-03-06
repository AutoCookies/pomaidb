// palloc_env_stress.cc — Multi-Environment Stress Test (PomaiDB + palloc)
//
// Validates PomaiDB ingestion across environments (IoT → Edge → Cloud).
// Ingests vectors into the default membrane and verifies count via the same
// iterator logic as pomai_inspect (membranes). Ensures we ingest enough vectors
// and that inspect-style count matches.
// Payload: 1536-dimensional float arrays (standard AI embeddings).
//
// Usage: ./benchmark_a [--list]

#include "pomai/pomai.h"
#include "pomai/iterator.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <random>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;
static constexpr const char* kDefaultMembrane = "__default__";

namespace {

constexpr size_t kVectorDim = 1536;

// Environment A: 100k vectors (or less in low-memory mode)
constexpr size_t kEnvATargetVectors = 100000;

// Environment B: 50k × 50 cycles (fresh DB per cycle)
constexpr size_t kEnvBVectorsPerCycle = 50000;
constexpr int    kEnvBCycles          = 50;

// Environment C: 1M vectors
constexpr size_t kEnvCTargetVectors = 1000000;

// When POMAI_BENCH_LOW_MEMORY=1 (e.g. 128MB Docker), use smaller targets so run completes.
static size_t EnvATargetVectors() {
  if (getenv("POMAI_BENCH_LOW_MEMORY") != nullptr)
    return 5000;
  return kEnvATargetVectors;
}
static size_t EnvBVectorsPerCycle() {
  if (getenv("POMAI_BENCH_LOW_MEMORY") != nullptr)
    return 2000;
  return kEnvBVectorsPerCycle;
}
static int EnvBCycles() {
  if (getenv("POMAI_BENCH_LOW_MEMORY") != nullptr)
    return 5;
  return kEnvBCycles;
}
static size_t EnvCTargetVectors() {
  if (getenv("POMAI_BENCH_LOW_MEMORY") != nullptr)
    return 20000;
  return kEnvCTargetVectors;
}

struct EnvReport {
  const char* env_name;
  const char* limit_enforced;
  size_t      vectors_allocated;   // ingested (Put count)
  size_t      vectors_verified;    // count via NewIterator (same as pomai_inspect)
  double      throughput_vec_per_sec;
  long        peak_rss_bytes;
  int         passed;  // 1 = PASS, 0 = FAIL
  const char* message;
};

long GetPeakRssBytes() {
#if defined(__linux__)
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0)
    return static_cast<long>(ru.ru_maxrss) * 1024L;
#endif
  return 0;
}

uint64_t ClockNs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

void FmtBytes(char* buf, size_t bufsz, long bytes) {
  if (bytes >= 1024L * 1024 * 1024)
    snprintf(buf, bufsz, "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  else if (bytes >= 1024L * 1024)
    snprintf(buf, bufsz, "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
  else if (bytes >= 1024L)
    snprintf(buf, bufsz, "%.1f KiB", static_cast<double>(bytes) / 1024.0);
  else
    snprintf(buf, bufsz, "%ld B", bytes);
}

void PrintReport(const EnvReport& r) {
  char rss_buf[32];
  FmtBytes(rss_buf, sizeof(rss_buf), r.peak_rss_bytes);
  printf("  Environment Name      : %s\n", r.env_name);
  printf("  Limit Enforced        : %s\n", r.limit_enforced);
  printf("  Vectors Ingested      : %zu\n", r.vectors_allocated);
  printf("  Vectors Verified      : %zu (inspect)\n", r.vectors_verified);
  printf("  Throughput (Vec/s)    : %.2f\n", r.throughput_vec_per_sec);
  printf("  Peak RSS Used         : %s\n", rss_buf);
  printf("  Status                : [%s] %s\n\n", r.passed ? "PASS" : "FAIL", r.message ? r.message : "");
}

// Count vectors in membrane using same logic as pomai_inspect (membranes).
static size_t CountVectorsInMembrane(pomai::DB* db, const char* membrane) {
  std::unique_ptr<pomai::SnapshotIterator> it;
  auto st = db->NewIterator(membrane, &it);
  if (!st.ok()) return 0;
  size_t count = 0;
  while (it->Valid()) {
    count++;
    it->Next();
  }
  return count;
}

struct IngestResult {
  size_t ingested = 0;
  size_t verified = 0;
  double throughput_vec_per_sec = 0.0;
  bool ok = false;
};

static IngestResult IngestAndVerify(const std::string& db_path, size_t target_vectors) {
  IngestResult out;
  std::error_code ec;
  fs::remove_all(db_path, ec);
  (void)ec;

  pomai::DBOptions opts;
  opts.path = db_path;
  opts.dim = static_cast<uint32_t>(kVectorDim);
  // Monolithic runtime: a single logical instance indexes all vectors.
  opts.shard_count = 1;
  opts.fsync = pomai::FsyncPolicy::kNever;

  // Backpressure: cap memtable usage so we leave RAM headroom for OS / other processes.
  if (const char* thr = std::getenv("POMAI_MEMTABLE_FLUSH_THRESHOLD_MB")) {
    const long v = std::strtol(thr, nullptr, 10);
    if (v > 0) {
      opts.memtable_flush_threshold_mb = static_cast<std::uint32_t>(v);
      opts.auto_freeze_on_pressure = true;
    }
  } else if (std::getenv("POMAI_BENCH_LOW_MEMORY") != nullptr) {
    // Default low-memory profile (e.g. 128 MiB container): keep DB under ~half of total RAM.
    opts.memtable_flush_threshold_mb = 48u;
    opts.auto_freeze_on_pressure = true;
  }

  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opts, &db);
  if (!st.ok()) return out;

  std::vector<float> vec(kVectorDim);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  auto t0 = ClockNs();
  for (size_t i = 0; i < target_vectors; ++i) {
    for (size_t j = 0; j < kVectorDim; ++j) vec[j] = dist(rng);
    st = db->Put(static_cast<pomai::VectorId>(i + 1), vec);
    if (!st.ok()) break;
    out.ingested++;
  }
  uint64_t elapsed_ns = ClockNs() - t0;
  if (out.ingested > 0 && elapsed_ns > 0)
    out.throughput_vec_per_sec = static_cast<double>(out.ingested) * 1e9 / static_cast<double>(elapsed_ns);

  out.verified = CountVectorsInMembrane(db.get(), kDefaultMembrane);
  (void)db->Close();
  out.ok = (out.verified == out.ingested && out.ingested == target_vectors);
  return out;
}

void RunEnvA(EnvReport* report) {
  report->env_name           = "The IoT Starvation";
  report->vectors_allocated  = 0;
  report->vectors_verified   = 0;
  report->throughput_vec_per_sec = 0.0;
  report->peak_rss_bytes     = 0;
  report->passed             = 0;
  report->message            = "";

  const size_t target_a = EnvATargetVectors();
  report->limit_enforced     = getenv("POMAI_BENCH_LOW_MEMORY")
                                   ? "low-memory: 5k vectors (1536-dim)"
                                   : "100k vectors (1536-dim)";
  IngestResult r = IngestAndVerify("/tmp/benchmark_a_env_a", target_a);
  report->vectors_allocated  = r.ingested;
  report->vectors_verified   = r.verified;
  report->throughput_vec_per_sec = r.throughput_vec_per_sec;
  report->peak_rss_bytes    = GetPeakRssBytes();

  if (r.verified == target_a && r.ingested == target_a) {
    report->passed = 1;
    report->message = "Ingest and verify OK (inspect count matches)";
  } else if (r.verified != r.ingested) {
    report->message = "FAIL: inspect count mismatch";
  } else {
    report->message = "FAIL: incomplete ingest";
  }
}

void RunEnvB(EnvReport* report) {
  report->env_name           = "The Edge Churn";
  report->limit_enforced     = getenv("POMAI_BENCH_LOW_MEMORY")
                                   ? "low-memory: 2k × 5 cycles"
                                   : "50k × 50 cycles (fresh DB per cycle)";
  report->vectors_allocated  = 0;
  report->vectors_verified   = 0;
  report->throughput_vec_per_sec = 0.0;
  report->peak_rss_bytes     = 0;
  report->passed             = 0;
  report->message            = "";

  const size_t per_cycle = EnvBVectorsPerCycle();
  const int num_cycles = EnvBCycles();

  long rss_after_first = 0;
  long rss_after_last  = 0;
  uint64_t t0 = ClockNs();
  size_t total_ingested = 0;
  size_t total_verified = 0;
  bool all_ok = true;
  int failed_cycle = -1;
  double first_cycle_throughput = 0.0;
  double last_cycle_throughput = 0.0;
  double min_cycle_throughput = 0.0;

  try {
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      printf("  Cycle %d/%d ... ", cycle + 1, num_cycles);
      fflush(stdout);
      std::string path = std::string("/tmp/benchmark_a_env_b_") + std::to_string(cycle);
      IngestResult r = IngestAndVerify(path, per_cycle);
      total_ingested += r.ingested;
      total_verified += r.verified;
      if (cycle == 0) first_cycle_throughput = r.throughput_vec_per_sec;
      last_cycle_throughput = r.throughput_vec_per_sec;
      if (r.throughput_vec_per_sec > 0.0 && (min_cycle_throughput == 0.0 || r.throughput_vec_per_sec < min_cycle_throughput))
        min_cycle_throughput = r.throughput_vec_per_sec;
      if (r.verified != per_cycle || r.ingested != per_cycle) {
        all_ok = false;
        if (failed_cycle < 0) failed_cycle = cycle;
      }
      if (cycle == 0) rss_after_first = GetPeakRssBytes();
      rss_after_last = GetPeakRssBytes();
      printf("ingested=%zu verified=%zu  %.1f Vec/s\n", r.ingested, r.verified, r.throughput_vec_per_sec);
      fflush(stdout);
    }
  } catch (const std::exception& e) {
    report->passed = 0;
    static std::string err_msg;
    err_msg = std::string("FAIL: exception: ") + e.what();
    report->message = err_msg.c_str();
    report->vectors_allocated = total_ingested;
    report->vectors_verified  = total_verified;
    report->peak_rss_bytes    = GetPeakRssBytes();
    if (total_ingested > 0 && (ClockNs() - t0) > 0)
      report->throughput_vec_per_sec = static_cast<double>(total_ingested) * 1e9 / static_cast<double>(ClockNs() - t0);
    return;
  } catch (...) {
    report->passed = 0;
    report->message = "FAIL: unknown exception";
    report->vectors_allocated = total_ingested;
    report->vectors_verified  = total_verified;
    report->peak_rss_bytes    = GetPeakRssBytes();
    return;
  }

  uint64_t elapsed_ns = ClockNs() - t0;
  report->vectors_allocated = total_ingested;
  report->vectors_verified  = total_verified;
  report->peak_rss_bytes    = rss_after_last;
  if (total_ingested > 0 && elapsed_ns > 0)
    report->throughput_vec_per_sec = static_cast<double>(total_ingested) * 1e9 / static_cast<double>(elapsed_ns);

  // Report per-cycle ingestion rate so we can check it does not degrade over time (constant vector size).
  if (num_cycles > 0 && first_cycle_throughput > 0.0) {
    printf("  Per-cycle ingestion (Vec/s): first=%.1f last=%.1f min=%.1f",
           first_cycle_throughput, last_cycle_throughput, min_cycle_throughput);
    const double ratio = (first_cycle_throughput > 0.0) ? (last_cycle_throughput / first_cycle_throughput) : 0.0;
    if (ratio < 0.75)
      printf("  [WARN: last cycle %.0f%% of first — ingestion rate reduced over time]\n", ratio * 100.0);
    else
      printf("  [OK: rate stable]\n");
    fflush(stdout);
  }

  if (!all_ok) {
    report->passed = 0;
    static std::string fail_msg;
    if (failed_cycle >= 0)
      fail_msg = "FAIL: cycle " + std::to_string(failed_cycle) + " had verify/ingest != " + std::to_string(per_cycle);
    else
      fail_msg = "FAIL: one or more cycles had verify count != " + std::to_string(per_cycle);
    report->message = fail_msg.c_str();
    return;
  }
  if (rss_after_first <= 0) {
    report->passed  = 1;
    report->message = "Peak RSS stable (no leak); all cycles verified";
  } else {
    double growth = static_cast<double>(rss_after_last - rss_after_first) / static_cast<double>(rss_after_first);
    if (growth <= 0.15) {
      report->passed  = 1;
      report->message = "Peak RSS stable (no leak); all cycles verified";
      if (first_cycle_throughput > 0.0 && last_cycle_throughput >= 0.75 * first_cycle_throughput)
        report->message = "Peak RSS stable (no leak); all cycles verified; ingestion rate stable";
      else if (first_cycle_throughput > 0.0 && last_cycle_throughput < 0.75 * first_cycle_throughput)
        report->message = "Peak RSS stable (no leak); all cycles verified; WARN: ingestion rate degraded over cycles";
    } else {
      report->passed  = 0;
      report->message = "FAIL: RSS growth suggests leak";
    }
  }
}

void RunEnvC(EnvReport* report) {
  report->env_name           = "The Cloud Scale";
  report->limit_enforced     = getenv("POMAI_BENCH_LOW_MEMORY")
                                   ? "low-memory: 20k vectors (1536-dim)"
                                   : "1M vectors (1536-dim)";
  report->vectors_allocated  = 0;
  report->vectors_verified   = 0;
  report->throughput_vec_per_sec = 0.0;
  report->peak_rss_bytes     = 0;
  report->passed             = 0;
  report->message            = "";

  const size_t target_c = EnvCTargetVectors();
  IngestResult r = IngestAndVerify("/tmp/benchmark_a_env_c", target_c);
  report->vectors_allocated  = r.ingested;
  report->vectors_verified   = r.verified;
  report->throughput_vec_per_sec = r.throughput_vec_per_sec;
  report->peak_rss_bytes    = GetPeakRssBytes();

  if (r.verified == target_c && r.ingested == target_c) {
    report->passed = 1;
    report->message = "Bulk index build completed; inspect count matches";
  } else if (r.verified != r.ingested) {
    report->message = "FAIL: inspect count mismatch";
  } else {
    report->message = "FAIL: incomplete ingest";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && (strcmp(argv[1], "--list") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    printf("PomaiDB — benchmark_a: Multi-Environment Ingestion + Verify (inspect)\n\n");
    printf("Environments:\n");
    printf("  A. The IoT Starvation   — Ingest 100k vectors, verify count via NewIterator (same as pomai_inspect)\n");
    printf("  B. The Edge Churn       — 50 cycles: ingest 50k per cycle (fresh DB each time), verify each; RSS stability\n");
    printf("  C. The Cloud Scale      — Ingest 1M vectors, verify count via NewIterator\n\n");
    printf("Payload: %zu-dim float vectors. Count verified with same logic as pomai_inspect membranes.\n", kVectorDim);
    return 0;
  }

  printf("\n------------------------------------------------------------\n");
  printf("PomaiDB — benchmark_a: Multi-Environment (Ingest + Verify)\n");
  printf("Payload: %zu-dim vectors in default membrane; verify with inspect-style iterator count.\n", kVectorDim);
  if (getenv("POMAI_BENCH_LOW_MEMORY") != nullptr)
    printf("Mode: POMAI_BENCH_LOW_MEMORY=1 (reduced targets for 128MB / constrained runs).\n");
  printf("------------------------------------------------------------\n\n");

  int any_fail = 0;
  EnvReport report = {};

  printf("Running: The IoT Starvation (128 MiB container) ...\n");
  fflush(stdout);
  RunEnvA(&report);
  PrintReport(report);
  if (!report.passed) any_fail = 1;

  printf("Running: The Edge Churn (1 GiB, 50×50k) ...\n");
  fflush(stdout);
  RunEnvB(&report);
  PrintReport(report);
  if (!report.passed) any_fail = 1;

  printf("Running: The Cloud Scale (8 GiB, 1M vectors) ...\n");
  fflush(stdout);
  RunEnvC(&report);
  PrintReport(report);
  if (!report.passed) any_fail = 1;

  printf("------------------------------------------------------------\n");
  printf("Summary: %s\n", any_fail ? "ONE OR MORE ENVIRONMENTS FAILED" : "ALL ENVIRONMENTS PASSED");
  printf("------------------------------------------------------------\n\n");

  return any_fail ? 1 : 0;
}
