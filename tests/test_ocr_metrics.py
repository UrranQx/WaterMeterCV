import csv

from models.metrics.ocr_metrics import (
    append_ocr_comparison_row,
    build_ocr_comparison_row,
    evaluate_ocr_batch,
    full_string_accuracy_mode,
    normalize_reading,
)


def test_normalize_reading_leading_zeros():
    assert normalize_reading("00482") == "482"
    assert normalize_reading("000") == "0"
    assert normalize_reading("0") == "0"


def test_full_string_accuracy_mode_raw_vs_normalized_differs():
    preds = ["482"]
    gts = ["00482"]
    raw = full_string_accuracy_mode(preds, gts, mode="raw")
    norm = full_string_accuracy_mode(preds, gts, mode="normalized")
    assert raw == 0.0
    assert norm == 1.0


def test_evaluate_ocr_batch_keys_and_ranges():
    preds = ["123", "129"]
    gts = ["123", "120"]
    metrics = evaluate_ocr_batch(preds, gts)

    assert set(metrics.keys()) == {"fsa_raw", "fsa_norm", "pda", "cer"}
    assert 0.0 <= metrics["fsa_raw"] <= 1.0
    assert 0.0 <= metrics["fsa_norm"] <= 1.0
    assert 0.0 <= metrics["pda"] <= 1.0
    assert metrics["cer"] >= 0.0


def test_build_and_append_ocr_comparison_row(tmp_path):
    poly = {"fsa_raw": 0.9, "fsa_norm": 0.95, "pda": 0.97, "cer": 0.03}
    bbox = {"fsa_raw": 0.8, "fsa_norm": 0.85, "pda": 0.9, "cer": 0.08}

    row = build_ocr_comparison_row(
        method="crnn_ctc",
        wm_poly_metrics=poly,
        wm_bbox_metrics=bbox,
        wm_poly_ms=12.5,
        wm_bbox_ms=14.0,
        run_date="2026-04-16",
    )

    csv_path = tmp_path / "ocr_comparison.csv"
    append_ocr_comparison_row(csv_path, row)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["method"] == "crnn_ctc"
    assert rows[0]["run_date"] == "2026-04-16"
