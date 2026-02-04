# PomaiDB Recall & Search Quality

This document describes the recall methodology, testing harness, and tuning parameters for PomaDB's search engine.

## Overview

PomaiDB targets **Recall@10 >= 0.95** for production search workloads. To ensure this, we maintain:
1.  **Ground Truth Oracle**: A standard brute-force implementation (`BruteForceSearch`) that computes exact distances and top-k ranking.
2.  **Recall Harness**: Use `recall_test` to generate deterministic synthetic datasets (Clustered and Uniform) and verify recall against the Oracle.
3.  **Strict Gates**: The CI process enforces recall targets.

## Methodology

### Indexing
We use `IvfCoarse` (Inverted File Index) for candidate coarse routing.
- **Training**: We use **KMeans++** initialization and batch KMeans training (`KMeansLite`) to ensure robust centroids regardless of insertion order.
- **Buffering**: Vectors are buffered until a sufficient number (default `nlist * 40`) is collected before training occurs.

### Search
- **Routing**: Query is compared against all `nlist` centroids. The top `nprobe` centroids are selected.
- **Fine Scan**: All vectors assigned to the selected centroids are scanned exactly (using SIMD-friendly Dot Product) and re-ranked.
- **Tie-Breaking**: If scores are identical, we break ties using `VectorId` (ascending) to guarantee determinism.

## Running Tests

To run the recall suite:

```bash
cd build
cmake -DPOMAI_BUILD_TESTS=ON ..
make recall_test
./recall_test
```

### Interpretation of Output

The test outputs a report:
```
RECALL REPORT
Dataset: 5000 vectors, 32 dim, 5 clusters
Top-K: 10
Mean Recall: 1
Min Recall: 1
Latency p50: 2000 us
Latency p95: 12000 us
```
- **Mean Recall**: Target >= 0.95.
- **Latency**: Ensure p95 is acceptable (typically < 50ms for embedded use cases, though this depends on dataset size).

## Tuning

If recall fails (e.g. on new datasets):
1.  **Increase `nprobe`**: This searches more buckets. Increases latency but improves recall.
2.  **Increase `nlist`**: Finer clusters. Reduces candidate set size (improves latency) but requires higher `nprobe` to maintain recall.
3.  **Check Training**: Ensure your dataset is representative. The `IvfCoarse` trainer uses the first batch of data to train.

## Troubleshooting

- **Low Recall**: Usually means `nprobe` is too low for the data distribution (e.g. Uniform data requires higher `nprobe`).
- **High Latency**: Reduce `nprobe` or increase `nlist`.
