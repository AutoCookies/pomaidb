#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_f32bin(path: Path, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(f)
        arr.tofile(f)


def gather_system_info():
    cpu_model = "unknown"
    mem_total = "unknown"
    os_version = platform.platform()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu_model": cpu_model, "ram": mem_total, "os_version": os_version}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def build_ground_truth(base: np.ndarray, queries: np.ndarray, k: int, metric: str):
    if metric == "l2":
        dists = np.sum((queries[:, None, :] - base[None, :, :]) ** 2, axis=2)
        return np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    if metric in ("ip", "cosine"):
        scores = queries @ base.T
        return np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    raise ValueError(f"Unsupported metric: {metric}")


def assert_exact_recall(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, metric: str, k: int = 10):
    import faiss

    if metric == "l2":
        idx = faiss.IndexFlatL2(base.shape[1])
        baseline_name = "faiss.IndexFlatL2"
    else:
        idx = faiss.IndexFlatIP(base.shape[1])
        baseline_name = "faiss.IndexFlatIP"
    idx.add(base)
    _, pred = idx.search(queries, k)

    hits = 0
    for i in range(pred.shape[0]):
        hits += len(set(pred[i, :k].tolist()).intersection(set(gt[i, :k].tolist())))
    recall = hits / float(pred.shape[0] * k)
    if recall < 0.999:
        raise RuntimeError(
            f"Ground-truth sanity check failed for metric={metric}: "
            f"{baseline_name} recall={recall:.6f} (expected >= 0.999)."
        )


def run_worker(py, worker, engine, out_json, dataset, queries, gt, metric, libpomai=None):
    cmd = [
        py,
        str(worker),
        "--engine",
        engine,
        "--dataset",
        str(dataset),
        "--queries",
        str(queries),
        "--ground-truth",
        str(gt),
        "--repeats",
        "3",
        "--output",
        str(out_json),
        "--metric",
        str(metric),
    ]
    if libpomai:
        cmd += ["--libpomai", str(libpomai)]
    subprocess.run(cmd, check=True)


def make_plots(results, out_dir: Path):
    engines = [r["engine"] for r in results]

    qps = [r["query_throughput_qps"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(engines, qps)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Queries per second")
    plt.title("Query Throughput (higher is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "qps_bar.png", dpi=150)
    plt.close()

    lat = [r["avg_latency_ms"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(engines, lat)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average latency (ms/query)")
    plt.title("Average Query Latency (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_bar.png", dpi=150)
    plt.close()

    mem = [r["peak_rss_mb"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(engines, mem)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Peak RSS (MB)")
    plt.title("Peak Memory Usage")
    plt.tight_layout()
    plt.savefig(out_dir / "memory_bar.png", dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="benchmarks/cross_engine/output")
    p.add_argument("--libpomai", default="build/libpomai_c.so")
    p.add_argument("--metric", choices=["l2", "ip", "cosine"], default="ip")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n, dim, nq = 100000, 128, 1000
    base = np.random.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    queries = np.random.uniform(0.0, 1.0, size=(nq, dim)).astype(np.float32)

    if args.metric == "cosine":
        base = normalize_rows(base)
        queries = normalize_rows(queries)

    dataset_path = out_dir / "dataset_base.f32bin"
    query_path = out_dir / "dataset_queries.f32bin"
    gt_path = out_dir / "ground_truth_top10.npy"
    save_f32bin(dataset_path, base)
    save_f32bin(query_path, queries)
    gt = build_ground_truth(base, queries, 10, args.metric)
    assert_exact_recall(base, queries, gt, args.metric, 10)
    np.save(gt_path, gt)

    worker = Path(__file__).with_name("engine_worker.py")
    py = sys.executable

    outputs = []
    engine_map = [
        ("pomai", {"libpomai": Path(args.libpomai)}),
        ("hnswlib", {}),
        ("faiss_flat", {}),
        ("faiss_hnsw", {}),
    ]

    for engine, extra in engine_map:
        out_json = out_dir / f"{engine}.json"
        run_worker(
            py,
            worker,
            engine,
            out_json,
            dataset_path,
            query_path,
            gt_path,
            args.metric,
            libpomai=extra.get("libpomai"),
        )
        with open(out_json, "r", encoding="utf-8") as f:
            outputs.append(json.load(f))

    skipped = []
    for optional in ["qdrant", "milvus"]:
        skipped.append({"engine": optional, "status": "skipped", "reason": "optional engine not configured in local environment"})

    system_info = gather_system_info()
    payload = {
        "seed": 42,
        "metric": args.metric,
        "dataset": {
            "vectors": n,
            "queries": nq,
            "dimension": dim,
            "distribution": "uniform[0,1]",
            "dtype": "float32",
            "normalized": args.metric == "cosine",
            "similarity": "ip" if args.metric in ("ip", "cosine") else "l2",
        },
        "system": system_info,
        "results": outputs,
        "skipped_optional": skipped,
        "commands": [
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
            "cmake --build build -j",
            f"python3 benchmarks/cross_engine/run_benchmark.py --output-dir benchmarks/cross_engine/output --libpomai build/libpomai_c.so --metric {args.metric}",
        ],
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    make_plots(outputs, out_dir)

    lines = [
        "# PomaiDB Cross-Engine Benchmark Results",
        "",
        "## Hardware / OS",
        f"- CPU: {system_info['cpu_model']}",
        f"- RAM: {system_info['ram']}",
        f"- OS: {system_info['os_version']}",
        "",
        "## Reproducibility",
        "- Seed: 42",
        "- Dataset: 100,000 base vectors, 1,000 query vectors, dim=128, float32, uniform [0,1]",
        "- K: 10",
        f"- Metric: {args.metric}",
        f"- Normalization applied: {args.metric == 'cosine'}",
        "",
        "## Commands Used",
    ]
    lines += [f"- `{c}`" for c in payload["commands"]]
    lines += [
        "",
        "## Engine Parameters",
    ]
    for r in outputs:
        lines.append(f"- **{r['engine']}**: `{json.dumps(r['params'])}`")

    lines += [
        "",
        "## Results",
        "",
        "| Engine | Ingestion (s) | Index build (s) | QPS | Avg latency (ms) | Peak RSS (MB) | Disk usage (MB) | Recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in outputs:
        lines.append(
            f"| {r['engine']} | {r['ingestion_time_s']:.3f} | {r['index_build_time_s']:.3f} | {r['query_throughput_qps']:.2f} | {r['avg_latency_ms']:.3f} | {r['peak_rss_mb']:.2f} | {r['disk_usage_bytes']/1024/1024:.2f} | {r['recall_at_10']:.4f} |"
        )

    lines += [
        "",
        "## Optional Engines",
    ]
    for s in skipped:
        lines.append(f"- {s['engine']}: {s['status']} ({s['reason']})")

    best_qps = max(outputs, key=lambda x: x["query_throughput_qps"])
    best_lat = min(outputs, key=lambda x: x["avg_latency_ms"])
    best_mem = min(outputs, key=lambda x: x["peak_rss_mb"])

    lines += [
        "",
        "## Analysis",
        f"- Exact baseline (Faiss {'IndexFlatL2' if args.metric == 'l2' else 'IndexFlatIP'}) provides recall=1.0 for the chosen metric.",
        f"- Metric used: **{args.metric}**.",
        f"- Input normalization applied: **{args.metric == 'cosine'}**.",
        f"- Fastest throughput: **{best_qps['engine']}** ({best_qps['query_throughput_qps']:.2f} QPS).",
        f"- Lowest latency: **{best_lat['engine']}** ({best_lat['avg_latency_ms']:.3f} ms/query).",
        f"- Lowest memory: **{best_mem['engine']}** ({best_mem['peak_rss_mb']:.2f} MB peak RSS).",
        "- Accuracy/performance tradeoff: exact methods provide strongest recall at higher compute cost; graph methods (hnswlib/Faiss HNSW/PomaiDB's current indexing path) trade some recall for speed and memory depending on parameters.",
        "- PomaiDB standing: compare its table row with graph-based peers and exact baseline to assess whether it is closer to high-recall or high-throughput operation under default safe durability settings.",
        "",
        "## Plot Artifacts",
        "- `qps_bar.png`",
        "- `latency_bar.png`",
        "- `memory_bar.png`",
    ]

    with open(out_dir / "results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
