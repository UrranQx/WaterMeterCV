"""Higher-level OCR metrics and comparison logging helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
import csv

from models.metrics.evaluation import (
    full_string_accuracy,
    per_digit_accuracy,
    character_error_rate,
)

OCR_COMPARISON_COLUMNS = [
    "method",
    "wm_poly_fsa_raw",
    "wm_poly_fsa_norm",
    "wm_poly_pda",
    "wm_poly_cer",
    "wm_poly_ms",
    "wm_bbox_fsa_raw",
    "wm_bbox_fsa_norm",
    "wm_bbox_pda",
    "wm_bbox_cer",
    "wm_bbox_ms",
    "run_date",
]


def normalize_reading(value: str) -> str:
    """Normalize meter reading for leading-zero-insensitive comparison."""
    stripped = value.strip()
    if stripped == "":
        return ""
    normalized = stripped.lstrip("0")
    return normalized if normalized else "0"


def _validate_lengths(predictions: Sequence[str], ground_truths: Sequence[str]) -> None:
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "predictions and ground_truths must have equal length: "
            f"{len(predictions)} != {len(ground_truths)}"
        )


def full_string_accuracy_mode(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    mode: str = "raw",
) -> float:
    """Compute FSA in raw or normalized mode."""
    _validate_lengths(predictions, ground_truths)

    if mode == "raw":
        return full_string_accuracy(list(predictions), list(ground_truths))
    if mode == "normalized":
        pred_norm = [normalize_reading(p) for p in predictions]
        gt_norm = [normalize_reading(g) for g in ground_truths]
        return full_string_accuracy(pred_norm, gt_norm)
    raise ValueError(f"Unsupported mode: {mode!r}. Use 'raw' or 'normalized'.")


def mean_per_digit_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Mean per-digit accuracy across all samples."""
    _validate_lengths(predictions, ground_truths)
    if len(predictions) == 0:
        return 0.0
    return sum(per_digit_accuracy(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)


def mean_character_error_rate(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Mean CER across all samples."""
    _validate_lengths(predictions, ground_truths)
    if len(predictions) == 0:
        return 0.0
    return sum(character_error_rate(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)


def evaluate_ocr_batch(predictions: Sequence[str], ground_truths: Sequence[str]) -> dict[str, float]:
    """Compute the canonical OCR metric bundle for a prediction batch."""
    return {
        "fsa_raw": full_string_accuracy_mode(predictions, ground_truths, mode="raw"),
        "fsa_norm": full_string_accuracy_mode(predictions, ground_truths, mode="normalized"),
        "pda": mean_per_digit_accuracy(predictions, ground_truths),
        "cer": mean_character_error_rate(predictions, ground_truths),
    }


def build_ocr_comparison_row(
    method: str,
    wm_poly_metrics: dict[str, float],
    wm_bbox_metrics: dict[str, float],
    wm_poly_ms: float,
    wm_bbox_ms: float,
    run_date: str | None = None,
) -> dict[str, float | str]:
    """Build one row for results/ocr_comparison.csv schema."""
    return {
        "method": method,
        "wm_poly_fsa_raw": wm_poly_metrics["fsa_raw"],
        "wm_poly_fsa_norm": wm_poly_metrics["fsa_norm"],
        "wm_poly_pda": wm_poly_metrics["pda"],
        "wm_poly_cer": wm_poly_metrics["cer"],
        "wm_poly_ms": wm_poly_ms,
        "wm_bbox_fsa_raw": wm_bbox_metrics["fsa_raw"],
        "wm_bbox_fsa_norm": wm_bbox_metrics["fsa_norm"],
        "wm_bbox_pda": wm_bbox_metrics["pda"],
        "wm_bbox_cer": wm_bbox_metrics["cer"],
        "wm_bbox_ms": wm_bbox_ms,
        "run_date": run_date or datetime.now().strftime("%Y-%m-%d"),
    }


def append_ocr_comparison_row(csv_path: Path, row: dict[str, float | str]) -> None:
    """Append one row to OCR comparison CSV, creating header if needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = OCR_COMPARISON_COLUMNS
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header:
            fieldnames = header

    normalized_row = {name: row.get(name, "") for name in fieldnames}
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


__all__ = [
    "OCR_COMPARISON_COLUMNS",
    "normalize_reading",
    "full_string_accuracy_mode",
    "mean_per_digit_accuracy",
    "mean_character_error_rate",
    "evaluate_ocr_batch",
    "build_ocr_comparison_row",
    "append_ocr_comparison_row",
]
