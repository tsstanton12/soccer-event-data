# Ball Model Evaluation Workflow

Use this after annotating the hard-frame batch in Roboflow and training a new
ball detector.

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

## Why This Matters

This evaluates the next ball model on the exact hard cases that currently break
possession: low-confidence balls, long detection gaps, gap-middle frames, and
suspicious jumps across several venues. If the new model improves this batch,
we can rerun ball paths, ball-player association, and possession chains with
more confidence.
