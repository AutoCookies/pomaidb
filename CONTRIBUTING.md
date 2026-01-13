# Contributing to PomaiDB

First off, thank you for considering contributing to PomaiDB! It's people like you who help us prove that high-performance AI is possible on commodity hardware.

PomaiDB is an open-source project with a "source-available" license (CC BY-NC-SA 4.0). We welcome contributions from everyone—whether you are fixing a typo, optimizing a SIMD loop, or proposing a new architectural feature.

## The Philosophy
Before writing code, please understand the core philosophy of PomaiDB:

Efficiency is Paramount: We build for the "Dell Latitude E5440" scenario. Every byte of RAM and every CPU cycle matters.

Zero Bloat: We avoid heavy external dependencies. We prefer writing a custom, optimized 100-line implementation over linking a 100MB library.

Concurrency First: The system relies on std::shared_mutex and atomic operations. Code must be thread-safe for high-concurrency read workloads.

## How to Contribute
Reporting Bugs
If you find a bug, please open an Issue using the following template:

Hardware: CPU model (important for SIMD/AVX2), RAM size.

OS: Linux distribution and kernel version.

Steps to Reproduce: Minimal SQL commands to trigger the crash or error.

Logs: Output from the server console.

Suggesting Enhancements
We love new ideas! If you want to add a feature (e.g., Metadata Filtering, Bulk Loaders):

Open an Issue titled [Feature Request] Name of feature.

Explain why this is needed and how you plan to implement it (High-level architecture).

Wait for feedback from the maintainers before writing code to ensure it aligns with the roadmap.

Pull Requests (PRs)
Fork the repo and create your branch from main.

If you've added code that should be tested, add tests.

Ensure your code builds without warnings.

Benchmark it: If your PR touches the hot path (Search/Insert), run benchmarks/benchmark_scientific.py. If performance drops significantly, the PR will likely be rejected unless there is a strong justification.

Format your code to match the existing style (K&R style braces, 4-space indentation).

## Development Environment
Requirements:

C++20 compliant compiler (GCC 10+ or Clang 11+).

CMake 3.10+.

Linux (preferred) or WSL.

Build Command:

Bash

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
## Coding Standards
Modern C++: Use C++20 features (concepts, std::span, etc.) where they improve readability or safety without sacrificing speed.

Memory Management: Prefer smart pointers (std::unique_ptr) for ownership. Use raw pointers only for non-owning views in tight loops.

Error Handling: Use Exceptions (std::runtime_error) for fatal startup errors. For runtime errors (e.g., query parsing), return descriptive error strings via the protocol.

Naming:

Classes: PascalCase (e.g., ShardArena)

Methods/Variables: snake_case (e.g., alloc_blob, write_head_)

Private members: suffix with _ (e.g., mutex_)

## Architectural Guidelines
If you are modifying the core engine (PomaiOrbit):

The "Membrance" Concept: Remember that PomaiDB is multi-tenant. Operations on one membrance must never block operations on another.

Locking Strategy:

Readers (Search/Get): Must be virtually lock-free or use std::shared_lock.

Writers (Insert): Use fine-grained locks.

Structure (Create/Drop): Use std::unique_lock on the global DB mutex.

SIMD: If writing SIMD code, always provide a scalar fallback for older CPUs. Use __builtin_cpu_supports checks.

## License and Copyright
By contributing your code to PomaiDB, you agree to license your contribution under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

You retain copyright to your contributions.

You grant the project the right to distribute your code under the CC BY-NC-SA 4.0 terms.

This ensures the project remains free for education and research forever, preventing commercial exploitation without giving back to the community.

Thank you for helping us democratize AI!

— The PomaiDB