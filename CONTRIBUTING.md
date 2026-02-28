# 🤝 Contributing to PomaiDB

Thank you for considering contributing to **PomaiDB**!  
We truly believe every good idea — no matter how small — can help turn PomaiDB into **the most stable, reliable, and performant embedded vector database for real-world Edge AI**.

PomaiDB is **production-capable for embedded use**: we document API/ABI stability and versioning ([docs/VERSIONING.md](docs/VERSIONING.md)), run sanitizer CI (ASan, UBSan, TSan), and include recovery edge-case tests (backpressure, bad storage). We are **extremely serious** about building something that lasts on constrained devices (phones, Raspberry Pi, Jetson, WASM, low-RAM IoT).  
We care deeply about **stability**, **correctness**, **battery life**, **crash safety**, **ARM64/NEON performance**, and **zero bloat**.

Your contribution — whether it's a tiny bug fix, a benchmark on new hardware, a performance tweak, documentation, or a bold new feature — is **genuinely valued**.  
We read every issue, every PR, and every comment with attention.

## We Especially Welcome These Kinds of Contributions

We maintain this prioritized list so contributors know where help is most needed right now:

### Stability & Correctness (Highest Priority)
- Crash / power-loss recovery improvements
- WAL / manifest / Freeze edge-case tests (battery die, SD card corruption, OOM) — we already have recovery tests for backpressure and bad/missing storage in CI
- Thread-safety / race-condition fixes in sharded MemTables or snapshots
- Memory leak / undefined behavior reports + fixes (Valgrind, ASan, UBSan) — **ASan and UBSan runs are in CI**
- Fuzz testing on input vectors / queries

### Edge Hardware & Performance
- Benchmarks on real devices: Raspberry Pi 5/4, Orange Pi, Jetson Nano, Android phones, old laptops
- NEON / SVE / AVX2 / AVX-512 tuned distance kernels
- Allocation profiling & fragmentation reduction (mimalloc tuning, custom allocators)
- Power consumption measurements during ingest / search / freeze
- WASM-specific optimizations & size reductions

### Quantization & Approximate Indexing
- Integrating scalar quantization (SQ4/SQ8) and product quantization (PQ/OPQ/IVFPQ)
- Activating & tuning IVF from the existing partial code
- Lightweight HNSW integration (low M, efConstruction 40–80)
- Recall vs speed vs memory trade-off tables

### Bindings & Usability
- **Python**: `pip install pomaidb` is supported (see [python/](python/) and [docs/PYTHON_API.md](docs/PYTHON_API.md)); pybind11 bindings for a richer API are welcome
- Go / Rust / Swift / Kotlin bindings
- Simple CLI tool (`pomai put`, `pomai search`, `pomai freeze`)
- Example apps: offline RAG notebook, on-device agent memory

### Documentation & Education
- Clearer explanations of Freeze semantics, membrane types, consistency guarantees
- Real-world use-case guides (personal RAG, sensor time-series, photo embeddings)
- Benchmark methodology & reproducible scripts
- Vietnamese / multilingual docs (we're based in Vietnam!)

### Testing & CI
- More unit / integration tests (especially Freeze → recovery flows)
- Cross-platform CI (Linux ARM64, macOS Apple Silicon, Windows MSVC)
- **Sanitizer CI**: ASan, UBSan, and TSan runs are enabled in GitHub Actions (see [.github/workflows/ci.yml](.github/workflows/ci.yml))

### Small but Impactful Wins
- Better error messages & status codes
- Logging improvements (structured, levels)
- CMake presets & developer scripts
- GitHub Actions workflow optimizations

## How to Contribute (Step-by-Step)

1. **Find or create an issue**
   - Look at [open issues](https://github.com/AutoCookies/pomaidb/issues)
   - If your idea is new → open an issue first (label it `idea`, `enhancement`, `performance`, `stability`, etc.)
   - We **always** discuss ideas before coding — this saves everyone time

2. **Fork & branch**
   ```bash
   git clone git@github.com:YOUR_USERNAME/pomaidb.git
   cd pomaidb
   git checkout -b fix/crash-on-low-battery
   ```

3. **Develop**
   - Follow existing style (clang-format, CMake conventions)
   - Add tests (GoogleTest in `/tests/`)
   - Run benchmarks before/after if performance-related

4. **Commit & push**
   - Conventional commits preferred (e.g. `fix: handle OOM during Freeze`, `feat: NEON L2 kernel`, `docs: add RAG example`)
   - Keep commits small & focused

5. **Open Pull Request**
   - Fill the PR template (we have one)
   - Link related issue(s)
   - Describe what you changed and why
   - Add before/after benchmarks if relevant

6. **Wait for review**
   - We usually respond within 1–3 days
   - We may ask for changes — it's normal and helpful

## Code of Conduct

We follow the **Contributor Covenant Code of Conduct** — be kind, respectful, patient.  
All participants (maintainers, contributors, users) are expected to follow it.

## Recognition

Every merged contributor gets:
- Shout-out in release notes
- Added to CONTRIBUTORS file
- Eternal gratitude from the Edge AI community

We especially celebrate people who help make PomaiDB **stable on real devices** — because that's what matters most.

## Questions?

Open an issue with label `question` or ping @AutoCookies on GitHub / Discord (link coming soon).

Thank you — your ideas and code are what will make PomaiDB the go-to embedded vector DB for 2026 and beyond.

Let's bring fast, private, local AI to every device.  