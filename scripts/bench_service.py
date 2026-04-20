"""Regression bench: run test split through the running service, compare with single-stage CSV.

Usage:
    python scripts/bench_service.py --url http://localhost:8000 --tag cpu
    python scripts/bench_service.py --url http://localhost:8000 --tag gpu

Writes a per-sample CSV + prints summary (exact match raw / normalized, consistency with single-stage).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "WaterMetricsDATA" / "ocr_crops" / "wm_bbox" / "test" / "labels.csv"
SINGLE = (
    REPO
    / "results"
    / "ocr_pretrained_failures_single_image_pipeline"
    / "ultralytics_yolo11m_baseline"
    / "none"
    / "test"
    / "wm_bbox"
    / "all_predictions_single_image_pipeline.csv"
)
IMAGES = REPO / "WaterMetricsDATA" / "waterMeterDataset" / "WaterMeters" / "images"


def load_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            sid = row["filename"].removesuffix(".png")
            out[sid] = row["label"]
    return out


def load_single_stage() -> dict[str, str]:
    out: dict[str, str] = {}
    with SINGLE.open() as f:
        for row in csv.DictReader(f):
            out[row["sample_id"]] = row["pred"]
    return out


def norm(s: str) -> str:
    return s.lstrip("0") or "0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    labels = load_labels()
    single = load_single_stage()

    sample_ids = sorted(labels.keys())
    if args.limit:
        sample_ids = sample_ids[: args.limit]

    out_csv = REPO / "results" / f"service_bench_{args.tag}.csv"
    out_csv.parent.mkdir(exist_ok=True)

    rows: list[dict[str, str | float]] = []
    t_start = time.perf_counter()
    session = requests.Session()
    for i, sid in enumerate(sample_ids, 1):
        image_path = IMAGES / f"{sid}.jpg"
        if not image_path.exists():
            print(f"[skip] {sid}: image not found")
            continue

        gt = labels[sid]
        single_pred = single.get(sid, "")

        with image_path.open("rb") as fh:
            t0 = time.perf_counter()
            resp = session.post(
                f"{args.url}/predict",
                files={"image": (image_path.name, fh, "image/jpeg")},
                timeout=30,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            print(f"[err]  {sid}: HTTP {resp.status_code} {resp.text[:80]}")
            continue
        body = resp.json()
        pred = body["digits"]
        conf = body["confidence"]

        rows.append({
            "sample_id": sid,
            "gt": gt,
            "docker_pred": pred,
            "docker_conf": conf,
            "single_pred": single_pred,
            "ok_raw": pred == gt,
            "ok_norm": norm(pred) == norm(gt),
            "docker_eq_single": pred == single_pred,
            "dt_ms": round(dt_ms, 2),
        })
        if i % 50 == 0:
            print(f"  [{i}/{len(sample_ids)}]")

    wall = time.perf_counter() - t_start
    n = len(rows)
    if n == 0:
        print("no rows")
        return 1

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_raw = sum(r["ok_raw"] for r in rows)
    ok_norm = sum(r["ok_norm"] for r in rows)
    eq_single = sum(r["docker_eq_single"] for r in rows)
    single_ok_raw = sum(r["single_pred"] == r["gt"] for r in rows)
    single_ok_norm = sum(norm(r["single_pred"]) == norm(r["gt"]) for r in rows if r["single_pred"])
    avg_dt = sum(r["dt_ms"] for r in rows) / n

    print(f"\n=== {args.tag.upper()} bench: {n} samples, wall={wall:.1f}s, avg inference={avg_dt:.1f} ms ===")
    print(f"docker exact match (raw):        {ok_raw}/{n}  = {ok_raw/n:.1%}")
    print(f"docker exact match (normalized): {ok_norm}/{n} = {ok_norm/n:.1%}")
    print(f"single exact match (raw):        {single_ok_raw}/{n}  = {single_ok_raw/n:.1%}")
    print(f"single exact match (normalized): {single_ok_norm}/{n} = {single_ok_norm/n:.1%}")
    print(f"docker vs single (same pred):    {eq_single}/{n}  = {eq_single/n:.1%}")
    print(f"report: {out_csv.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
