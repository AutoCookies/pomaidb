#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import pomaidb


def http_get(url: str, token: str | None = None) -> str:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_post(url: str, body: str = "", token: str | None = None, *, idempotency: str | None = None) -> str:
    data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if idempotency:
        req.add_header("Idempotency-Key", idempotency)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def cmd_preflight(args) -> int:
    txt = pomaidb.resolve_effective_options(
        args.path,
        args.dim,
        profile=args.profile,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        adaptive_threshold=args.adaptive_threshold,
    )
    print(txt)
    return 0


def cmd_status(args) -> int:
    print(http_get(f"http://{args.host}:{args.port}/v1/health", args.token))
    print(http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token))
    return 0


def cmd_metrics(args) -> int:
    print(http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token))
    return 0


def cmd_doctor(args) -> int:
    health = http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token)
    metrics = http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token)
    print("healthz:", health)
    lines = [ln for ln in metrics.splitlines() if ln.strip()]
    wanted = ("http_errors_total", "ingest_errors_total", "rate_limited_total", "auth_denied_total")
    print("key_metrics:")
    for ln in lines:
        if any(ln.startswith(w) for w in wanted):
            print(" ", ln)
    return 0

def _open_local_db(args):
    return pomaidb.open_db(
        args.path,
        args.dim,
        shards=args.shards,
        metric=args.metric,
        profile=args.profile,
    )

def cmd_membranes(args) -> int:
    db = _open_local_db(args)
    try:
        names = pomaidb.list_membranes(db)
        print(json.dumps({"count": len(names), "membranes": names}))
    finally:
        pomaidb.close(db)
    return 0

def cmd_compact(args) -> int:
    db = _open_local_db(args)
    try:
        pomaidb.compact_membrane(db, args.membrane)
    finally:
        pomaidb.close(db)
    print(json.dumps({"status": "ok", "action": "compact", "membrane": args.membrane}))
    return 0

def cmd_gateway(args) -> int:
    health = http_get(f"http://{args.host}:{args.port}/v1/health", args.token)
    healthz = http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token)
    metrics = http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token)
    out = {
        "health": json.loads(health),
        "healthz": json.loads(healthz),
        "metrics_sample": {},
    }
    wanted = (
        "http_requests_total",
        "http_errors_total",
        "ingest_requests_total",
        "ingest_errors_total",
        "rate_limited_total",
        "auth_denied_total",
        "sync_attempts_total",
        "sync_success_total",
        "sync_fail_total",
        "sync_queue_depth",
        "sync_backlog_drops_total",
    )
    for ln in metrics.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0] in wanted:
            out["metrics_sample"][parts[0]] = parts[1]
    print(json.dumps(out))
    return 0

def cmd_ingest_http(args) -> int:
    """POST raw body to a gateway ingest path (e.g. /v1/ingest/vector/default/1)."""
    url = f"http://{args.host}:{args.port}{args.path}"
    body = ""
    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")
    elif args.body is not None:
        body = args.body
    out = http_post(url, body, args.token, idempotency=args.idempotency)
    print(out)
    return 0


def cmd_verify(args) -> int:
    db_path = Path(args.path)
    checks = {
        "db_path_exists": db_path.exists(),
        "manifest_exists": (db_path / "MANIFEST").exists(),
        "gateway_dir_exists": (db_path / "gateway").exists(),
    }
    ok = all(checks.values())
    print(json.dumps({"status": "ok" if ok else "degraded", "checks": checks}))
    return 0

def cmd_lifecycle(args) -> int:
    db = _open_local_db(args)
    try:
        if args.show:
            info = pomaidb.get_membrane_retention(db, args.membrane)
            print(json.dumps({"status": "ok", "action": "show", **info}))
            return 0
        if args.create:
            pomaidb.create_membrane_kind(
                db, args.membrane, args.dim, args.shards, args.kind,
                ttl_sec=args.ttl_sec,
                retention_max_count=args.retention_max_count,
                retention_max_bytes=args.retention_max_bytes,
            )
            action = "created"
        else:
            pomaidb.update_membrane_retention(
                db, args.membrane,
                ttl_sec=args.ttl_sec,
                retention_max_count=args.retention_max_count,
                retention_max_bytes=args.retention_max_bytes,
            )
            action = "updated"
    finally:
        pomaidb.close(db)
    print(json.dumps({
        "status": "ok",
        "action": action,
        "membrane": args.membrane,
        "ttl_sec": args.ttl_sec,
        "retention_max_count": args.retention_max_count,
        "retention_max_bytes": args.retention_max_bytes,
    }))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="pomaictl edge operations helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--path", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--profile", default="user_defined")
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--adaptive-threshold", type=int, default=0)
    p.set_defaults(func=cmd_preflight)

    for name, fn in (("status", cmd_status), ("metrics", cmd_metrics), ("doctor", cmd_doctor)):
        c = sub.add_parser(name)
        c.add_argument("--host", default="127.0.0.1")
        c.add_argument("--port", type=int, default=8080)
        c.add_argument("--token", default=None)
        c.set_defaults(func=fn)

    ing = sub.add_parser("ingest-http", help="POST to /v1/ingest/... (body from --body or --body-file)")
    ing.add_argument("--host", default="127.0.0.1")
    ing.add_argument("--port", type=int, default=8080)
    ing.add_argument("--token", default=None)
    ing.add_argument("--path", required=True, help="e.g. /v1/ingest/graph/vertex/default/10/1")
    ing.add_argument("--body", default=None, help="Request body (e.g. CSV floats)")
    ing.add_argument("--body-file", default=None, help="Read body from file")
    ing.add_argument("--idempotency", default=None, help="Idempotency-Key header value")
    ing.set_defaults(func=cmd_ingest_http)

    g = sub.add_parser("gateway")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--token", default=None)
    g.set_defaults(func=cmd_gateway)

    m = sub.add_parser("membranes")
    m.add_argument("--path", required=True)
    m.add_argument("--dim", type=int, required=True)
    m.add_argument("--shards", type=int, default=1)
    m.add_argument("--metric", default="ip")
    m.add_argument("--profile", default="user_defined")
    m.set_defaults(func=cmd_membranes)

    c = sub.add_parser("compact")
    c.add_argument("--path", required=True)
    c.add_argument("--dim", type=int, required=True)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--metric", default="ip")
    c.add_argument("--profile", default="user_defined")
    c.add_argument("--membrane", required=True)
    c.set_defaults(func=cmd_compact)

    v = sub.add_parser("verify")
    v.add_argument("--path", required=True)
    v.set_defaults(func=cmd_verify)

    l = sub.add_parser("lifecycle")
    l.add_argument("--path", required=True)
    l.add_argument("--dim", type=int, required=True)
    l.add_argument("--shards", type=int, default=1)
    l.add_argument("--metric", default="ip")
    l.add_argument("--profile", default="user_defined")
    l.add_argument("--membrane", required=True)
    l.add_argument("--ttl-sec", type=int, default=0)
    l.add_argument("--retention-max-count", type=int, default=0)
    l.add_argument("--retention-max-bytes", type=int, default=0)
    l.add_argument("--create", action="store_true", help="Create membrane with retention policy if missing")
    l.add_argument("--show", action="store_true", help="Show current retention policy for membrane")
    l.add_argument("--kind", type=int, default=12, help="Membrane kind when --create is used (default: 12/meta)")
    l.set_defaults(func=cmd_lifecycle)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
