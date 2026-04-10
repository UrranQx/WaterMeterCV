# Remaining Cells — yolo_single_stage.ipynb

## Current State

The notebook already has 4 cells (indices 0–3), committed at `045733e`:

| Index | Type     | Content                                      |
|-------|----------|----------------------------------------------|
| 0     | markdown | Title: "01 — Baseline: YOLO Single-Stage..." |
| 1     | code     | Setup: imports, ROOT detection, sys.path     |
| 2     | code     | Config: MODEL_SIZE, paths, hyperparams       |
| 3     | code     | Dataset verification: data.yaml fix, image count |

**Notebook format:** nbformat 4, nbformat_minor 4, python3 kernel, Python 3.13.

---

## Cells to Add (append in order)

### Cell 4 — Markdown: Training header

```markdown
## Training
```

---

### Cell 5 — Code: Training

```python
model = YOLO(f"{MODEL_SIZE}.pt")

results = model.train(
    data=str(FIXED_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=str(WEIGHTS_DIR),
    name=RUN_NAME,
    device=DEVICE,
    patience=PATIENCE,
    save=True,
)

print(f"\nTraining complete. Best weights: {WEIGHTS_DIR / RUN_NAME / 'weights' / 'best.pt'}")
```

**Commit after this cell:**
```
feat: baseline YOLO notebook — training cell
```

---

### Cell 6 — Markdown: Evaluation header

```markdown
## Evaluation
```

---

### Cell 7 — Code: Evaluation + Inference Time

```python
# Load best model
best_weights = WEIGHTS_DIR / RUN_NAME / "weights" / "best.pt"
best_model = YOLO(str(best_weights))

# ── YOLO built-in validation → mAP ──────────────────────────────
val_results = best_model.val(data=str(FIXED_YAML), split="test")
mAP50 = val_results.box.map50
mAP50_95 = val_results.box.map

print(f"mAP50:    {mAP50:.4f}")
print(f"mAP50-95: {mAP50_95:.4f}")
print(f"Per-class AP50:")
for i, name in enumerate(data_config["names"]):
    if i < len(val_results.box.ap50):
        print(f"  {name}: {val_results.box.ap50[i]:.4f}")


def predict_value(model, image_path):
    """Run YOLO and reconstruct digit string from detections.

    Returns (predicted_string, raw_result).
    """
    result = model.predict(str(image_path), verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        digit_mask = result.boxes.cls <= 9
        digit_boxes = result.boxes[digit_mask]
        if len(digit_boxes) > 0:
            sorted_idx = digit_boxes.xywh[:, 0].argsort()
            pred_str = "".join(str(int(digit_boxes.cls[i].item())) for i in sorted_idx)
            return pred_str, result
    return "", result


# ── Custom metrics + inference timing ────────────────────────────
test_samples = load_utility_meter_dataset(DATASET_PATH, split="test")

predictions = []
ground_truths = []

t_start = time.perf_counter()
for sample in test_samples:
    pred_str, _ = predict_value(best_model, sample.image_path)
    predictions.append(pred_str)

    gt_str = ""
    if sample.value is not None:
        gt_str = str(int(sample.value)) if sample.value == int(sample.value) else str(sample.value)
    ground_truths.append(gt_str)
t_total_ms = (time.perf_counter() - t_start) * 1000
avg_inference_ms = t_total_ms / len(test_samples)

# Compute metrics
fsa = full_string_accuracy(predictions, ground_truths)

pda_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if g]
pda = sum(per_digit_accuracy(p, g) for p, g in pda_pairs) / len(pda_pairs) if pda_pairs else 0.0

cer_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if g]
cer = sum(character_error_rate(p, g) for p, g in cer_pairs) / len(cer_pairs) if cer_pairs else 0.0

combined = 0.8 * mAP50 + 0.2 * fsa

print(f"\n{'='*50}")
print(f"Full-string accuracy: {fsa:.4f}")
print(f"Per-digit accuracy:   {pda:.4f}")
print(f"CER:                  {cer:.4f}")
print(f"Avg inference:        {avg_inference_ms:.1f} ms/image")
print(f"{'='*50}")
print(f"Combined Score (0.8×mAP50 + 0.2×FSA): {combined:.4f}")
```

**Commit after this cell:**
```
feat: baseline YOLO notebook — evaluation + inference timing
```

---

### Cell 8 — Markdown: Predictions header

```markdown
## Predictions
```

---

### Cell 9 — Code: Visualization (8-image grid)

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sample in zip(axes.flat, test_samples[:8]):
    img = cv2.imread(str(sample.image_path))
    if img is None:
        ax.set_title("not found")
        ax.axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    pred_str, result = predict_value(best_model, sample.image_path)

    # Draw predicted digit bboxes
    if result.boxes is not None and len(result.boxes) > 0:
        digit_mask = result.boxes.cls <= 9
        digit_boxes = result.boxes[digit_mask]
        if len(digit_boxes) > 0:
            bboxes = []
            for i in range(len(digit_boxes)):
                cls_id = int(digit_boxes.cls[i].item())
                cx = digit_boxes.xywh[i, 0].item() / w_img
                cy = digit_boxes.xywh[i, 1].item() / h_img
                bw = digit_boxes.xywh[i, 2].item() / w_img
                bh = digit_boxes.xywh[i, 3].item() / h_img
                bboxes.append((cls_id, cx, cy, bw, bh))
            img = draw_digit_bboxes(img, bboxes)

    gt_str = ""
    if sample.value is not None:
        gt_str = str(int(sample.value)) if sample.value == int(sample.value) else str(sample.value)

    ax.imshow(img)
    ax.set_title(f"GT={gt_str or '—'} | Pred={pred_str or '—'}", fontsize=10)
    ax.axis("off")

plt.suptitle(f"Baseline YOLO ({MODEL_SIZE}) — Test Predictions", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_predictions.png", dpi=150)
plt.close()
print("Saved to results/baseline_predictions.png")
```

---

### Cell 10 — Markdown: Cross-Dataset header

```markdown
## Cross-Dataset Evaluation (waterMeterDataset)

Per-digit accuracy only — full-string skipped because WM ground truth has decimals
(e.g. `78.677`) while YOLO predicts digit sequences only (`78677`). Decimal handling deferred.
```

---

### Cell 11 — Code: Cross-Dataset Evaluation

```python
wm_samples = load_water_meter_dataset(WM_PATH)

wm_pda_scores = []
for sample in wm_samples:
    pred_str, _ = predict_value(best_model, sample.image_path)

    # Strip decimal point from GT for digit-only comparison
    gt_str = ""
    if sample.value is not None:
        gt_str = str(sample.value).replace(".", "")

    if gt_str:
        wm_pda_scores.append(per_digit_accuracy(pred_str, gt_str))

wm_pda = sum(wm_pda_scores) / len(wm_pda_scores) if wm_pda_scores else 0.0
print(f"waterMeterDataset — per-digit accuracy: {wm_pda:.4f}  (N={len(wm_pda_scores)})")
```

---

### Cell 12 — Markdown: Save Results header

```markdown
## Save Results
```

---

### Cell 13 — Code: Save Results (JSON + CSV)

```python
# Save full metrics to JSON
metrics = {
    "model_size": MODEL_SIZE,
    "run_name": RUN_NAME,
    "primary_eval": {
        "mAP50": round(float(mAP50), 4),
        "mAP50_95": round(float(mAP50_95), 4),
        "full_string_accuracy": round(fsa, 4),
        "per_digit_accuracy": round(pda, 4),
        "CER": round(cer, 4),
        "combined_score": round(combined, 4),
        "avg_inference_ms": round(avg_inference_ms, 1),
    },
    "cross_dataset_eval": {
        "wm_per_digit_accuracy": round(wm_pda, 4),
    },
    "config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "patience": PATIENCE,
    },
    "run_date": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Append to comparison CSV (one row per model size run)
csv_path = RESULTS_DIR / "baseline_comparison.csv"
csv_exists = csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow([
            "model_size", "mAP50", "mAP50_95", "full_string_acc",
            "per_digit_acc", "CER", "combined_score", "inference_ms", "run_date",
        ])
    writer.writerow([
        MODEL_SIZE,
        round(float(mAP50), 4), round(float(mAP50_95), 4),
        round(fsa, 4), round(pda, 4), round(cer, 4),
        round(combined, 4), round(avg_inference_ms, 1),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ])

print(f"Metrics → {RESULTS_DIR / 'baseline_metrics.json'}")
print(f"CSV    → {csv_path}")
```

**Commit after cells 8–13:**
```
feat: baseline YOLO notebook — visualization, cross-dataset eval, results saving
```

---

### Cell 14 — Markdown: Conclusions

```markdown
## Conclusions

*(Fill after running)*

- **Model:** yolo11n / yolo11s / yolo11m
- **Combined Score:** ...
- **mAP50:** ...
- **Full-string accuracy:** ...
- **Per-digit accuracy:** ...
- **CER:** ...
- **Inference:** ... ms/image
- **Cross-dataset (WM per-digit):** ...
- **Next step:** upgrade model / proceed to ROI experiments
```

---

## Summary of Commits

| Commit | Cells added |
|--------|-------------|
| `feat: baseline YOLO notebook — training cell` | 4 (markdown), 5 (code) |
| `feat: baseline YOLO notebook — evaluation + inference timing` | 6 (markdown), 7 (code) |
| `feat: baseline YOLO notebook — visualization, cross-dataset eval, results saving` | 8–14 |

All cells are appended to the end of the existing notebook in order. No existing cells (0–3) are modified.
