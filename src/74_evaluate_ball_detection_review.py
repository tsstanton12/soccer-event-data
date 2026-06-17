#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


IMAGE_COLUMNS = ["image_path", "clean_image_path", "filename", "file_name", "image"]


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def read_image_size(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def load_yolo_boxes(label_path, image_width, image_height):
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _class_id, xc, yc, w, h = parts[:5]
        xc = float(xc) * image_width
        yc = float(yc) * image_height
        w = float(w) * image_width
        h = float(h) * image_height
        boxes.append([
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2,
        ])
    return boxes


def manifest_image_stem(row):
    return Path(row["clean_image_path"]).stem


def prediction_key_columns(predictions):
    for column in IMAGE_COLUMNS:
        if column in predictions.columns:
            return "image", column
    if {"venue", "frame"}.issubset(predictions.columns):
        return "venue_frame", None
    raise ValueError(
        "Predictions CSV must contain an image/filename column or venue+frame columns."
    )


def build_prediction_lookup(predictions, confidence_threshold):
    predictions = predictions.copy()
    if "object_type" in predictions.columns:
        predictions = predictions[predictions["object_type"].astype(str).str.lower().eq("ball")]
    if "class" in predictions.columns:
        predictions = predictions[predictions["class"].astype(str).str.lower().eq("ball")]
    if "label" in predictions.columns:
        predictions = predictions[predictions["label"].astype(str).str.lower().eq("ball")]
    if "confidence" in predictions.columns:
        predictions = predictions[predictions["confidence"] >= confidence_threshold]
    else:
        predictions["confidence"] = 1.0

    required = ["x1", "y1", "x2", "y2"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions CSV missing box columns: {missing}")

    key_type, image_column = prediction_key_columns(predictions)
    lookup = {}
    for _, row in predictions.iterrows():
        if key_type == "image":
            key = Path(str(row[image_column])).stem
        else:
            key = (str(row["venue"]), int(row["frame"]))
        lookup.setdefault(key, []).append(row)
    return lookup, key_type


def predictions_for_manifest_row(row, prediction_lookup, key_type):
    if key_type == "image":
        return prediction_lookup.get(manifest_image_stem(row), [])
    return prediction_lookup.get((str(row["venue"]), int(row["frame"])), [])


def evaluate(manifest, labels_dir, predictions, output_dir, iou_threshold, confidence_threshold):
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_lookup, key_type = build_prediction_lookup(predictions, confidence_threshold)
    rows = []

    for _, manifest_row in manifest.iterrows():
        image_path = Path(manifest_row["clean_image_path"])
        width, height = read_image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = load_yolo_boxes(label_path, width, height)
        pred_rows = predictions_for_manifest_row(manifest_row, prediction_lookup, key_type)
        pred_boxes = [
            [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            for row in pred_rows
        ]
        pred_confidences = [
            float(row.get("confidence", 1.0))
            for row in pred_rows
        ]

        best_iou = 0.0
        best_confidence = None
        for gt in gt_boxes:
            for pred_box, confidence in zip(pred_boxes, pred_confidences):
                iou = box_iou(gt, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_confidence = confidence

        gt_visible = len(gt_boxes) > 0
        predicted = len(pred_boxes) > 0

        if gt_visible and best_iou >= iou_threshold:
            outcome = "true_positive"
        elif gt_visible and predicted:
            outcome = "poor_localization"
        elif gt_visible:
            outcome = "missed_ball"
        elif predicted:
            outcome = "false_positive_on_null"
        else:
            outcome = "true_negative_null"

        rows.append({
            **manifest_row.to_dict(),
            "gt_box_count": len(gt_boxes),
            "prediction_count": len(pred_boxes),
            "best_iou": best_iou,
            "best_confidence": best_confidence,
            "outcome": outcome,
        })

    evaluated = pd.DataFrame(rows)
    detail_path = output_dir / "ball_detection_review_evaluation_details.csv"
    evaluated.to_csv(detail_path, index=False)

    summary = (
        evaluated.groupby(["venue", "reason", "outcome"])
        .size()
        .reset_index(name="count")
        .sort_values(["venue", "reason", "outcome"])
    )
    summary_path = output_dir / "ball_detection_review_evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    visible = evaluated[evaluated["gt_box_count"] > 0]
    nulls = evaluated[evaluated["gt_box_count"] == 0]
    recall = (
        (visible["outcome"] == "true_positive").mean()
        if len(visible)
        else None
    )
    null_fp_rate = (
        (nulls["outcome"] == "false_positive_on_null").mean()
        if len(nulls)
        else None
    )
    poor_localization_rate = (
        (visible["outcome"] == "poor_localization").mean()
        if len(visible)
        else None
    )

    report_lines = [
        "# Ball Detection Review Evaluation",
        "",
        f"Manifest rows: {len(evaluated)}",
        f"Visible-ball frames: {len(visible)}",
        f"Null/no-ball frames: {len(nulls)}",
        f"IoU threshold: {iou_threshold}",
        f"Confidence threshold: {confidence_threshold}",
        "",
        f"Recall on visible frames: {recall:.3f}" if recall is not None else "Recall on visible frames: n/a",
        f"False-positive rate on null frames: {null_fp_rate:.3f}" if null_fp_rate is not None else "False-positive rate on null frames: n/a",
        f"Poor-localization rate on visible frames: {poor_localization_rate:.3f}" if poor_localization_rate is not None else "Poor-localization rate on visible frames: n/a",
        "",
        "Outcome counts:",
        "",
        evaluated["outcome"].value_counts().to_string(),
        "",
        "Outputs:",
        "",
        f"- `{detail_path}`",
        f"- `{summary_path}`",
    ]
    report_path = output_dir / "ball_detection_review_evaluation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print("BALL DETECTION REVIEW EVALUATION COMPLETE")
    print("-----------------------------------------")
    print(f"Rows: {len(evaluated)}")
    print(f"Details: {detail_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print("")
    print(evaluated["outcome"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ball detections against Roboflow/YOLO annotations for "
            "the hard-frame review batch."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory of YOLO .txt annotation files matching clean-frame stems.",
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help=(
            "Model predictions CSV. Must include x1,y1,x2,y2 and either an "
            "image/filename column or venue+frame columns."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--confidence-threshold", type=float, default=0.20)
    args = parser.parse_args()

    evaluate(
        manifest=pd.read_csv(args.manifest),
        labels_dir=args.labels_dir,
        predictions=pd.read_csv(args.predictions_csv),
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
