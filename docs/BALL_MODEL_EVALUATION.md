# Ball Model Evaluation Workflow

Use this after annotating the hard-frame batch in Roboflow and training a new
ball detector.

## One-Command Handoff

After Roboflow annotation, train/export the new model. If the YOLO labels export
is ready, run the full validation/evaluation handoff:

```bash
.venv/bin/python src/82_run_post_roboflow_handoff.py \
  --model path/to/roboflow_best.pt \
  --labels-dir path/to/roboflow/export/labels \
  --full-clips lemoyne,army
```

If the labels export is not ready yet, run model-only overlay mode first:

```bash
.venv/bin/python src/82_run_post_roboflow_handoff.py \
  --model path/to/roboflow_best.pt \
  --full-clips lemoyne,army
```

For the latest reported Colab artifact, replace the model path with
`path/to/best_june_17.pt`.

Add `--dry-run` first to preview the planned commands. If an old hard-frame
evaluation exists, add:

```bash
--old-evaluation outputs/ball_detection_review/old_model_evaluation/ball_detection_review_evaluation_details.csv
```

With labels, the handoff runner validates the export, runs the model on the 750
hard frames, evaluates the result, optionally compares old vs new model
performance, and can rerun full clips through the stability-gated possession
overlay workflow. Without labels, it skips hard-frame validation/evaluation and
uses the model to generate full-clip rerun overlays.

## Training Batch

Upload only the clean frames:

```text
outputs/ball_detection_review/multi_venue_750/clean_frames/
```

Reference-only annotated previews:

```text
outputs/ball_detection_review/multi_venue_750/annotated_frames/
```

Keep one class:

```text
ball
```

If the ball is visible, draw one tight box. If no ball is visible, mark the
image as null/no object. Skip ambiguous frames rather than guessing.

## Export Ground Truth Labels

After annotation, export the batch from Roboflow in YOLO format. The evaluator
expects one `.txt` file per image, matching the clean-frame image stem.

Example:

```text
0001_army_low_confidence_detection_f001614_t0064.56.txt
```

Empty or missing `.txt` files are treated as no-ball/null frames.

Validate the export before training/evaluation:

```bash
.venv/bin/python src/80_validate_roboflow_ball_export.py \
  --manifest outputs/ball_detection_review/multi_venue_750/ball_detection_review_manifest.csv \
  --labels-dir path/to/roboflow/export/labels \
  --images-dir outputs/ball_detection_review/multi_venue_750/clean_frames \
  --output-dir outputs/ball_detection_review/roboflow_export_validation
```

The validator checks image coverage, YOLO label shape, class IDs, valid box
coordinates, visible-ball/null counts, and venue/reason breakdowns.

## Evaluate A Model

Run the new model on the same clean-frame images and save predictions as a CSV.
This helper writes the expected format:

```bash
.venv/bin/python src/75_detect_ball_review_images.py \
  --model path/to/new/roboflow_model.pt \
  --image-dir outputs/ball_detection_review/multi_venue_750/clean_frames \
  --output-csv outputs/ball_detection_review/new_model_predictions.csv \
  --conf 0.20
```

The predictions CSV contains:

```text
image_path or filename
x1
y1
x2
y2
confidence
```

Then evaluate:

```bash
.venv/bin/python src/74_evaluate_ball_detection_review.py \
  --manifest outputs/ball_detection_review/multi_venue_750/ball_detection_review_manifest.csv \
  --labels-dir path/to/roboflow/export/labels \
  --predictions-csv outputs/ball_detection_review/new_model_predictions.csv \
  --output-dir outputs/ball_detection_review/new_model_evaluation
```

The evaluator reports:

- recall on visible-ball frames;
- false-positive rate on null/no-ball frames;
- poor localization rate;
- results by venue and hard-case reason.

## Compare Old And New Models

After evaluating two models on the same Roboflow export, compare the detail
CSVs:

```bash
.venv/bin/python src/81_compare_ball_model_evaluations.py \
  --old-evaluation outputs/ball_detection_review/old_model_evaluation/ball_detection_review_evaluation_details.csv \
  --new-evaluation outputs/ball_detection_review/new_model_evaluation/ball_detection_review_evaluation_details.csv \
  --output-dir outputs/ball_detection_review/old_vs_new_model_comparison \
  --old-name current_model \
  --new-name roboflow_retrain
```

This reports improved/regressed/unchanged hard frames plus metric deltas by
venue and hard-case reason.

## Why This Matters

This evaluates the next ball model on the exact hard cases that currently break
possession: low-confidence balls, long detection gaps, gap-middle frames, and
suspicious jumps across several venues. If the new model improves this batch,
we can rerun ball paths, ball-player association, and possession chains with
more confidence.
