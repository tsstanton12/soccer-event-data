#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_image_size(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def parse_yolo_label(label_path):
    rows = []
    if not label_path.exists():
        return rows
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        row = {
            "line_number": line_number,
            "raw_line": stripped,
            "parse_error": "",
            "class_id": None,
            "x_center": None,
            "y_center": None,
            "width": None,
            "height": None,
        }
        if len(parts) < 5:
            row["parse_error"] = "too_few_columns"
            rows.append(row)
            continue
        try:
            row["class_id"] = int(float(parts[0]))
            row["x_center"] = float(parts[1])
            row["y_center"] = float(parts[2])
            row["width"] = float(parts[3])
            row["height"] = float(parts[4])
        except ValueError:
            row["parse_error"] = "non_numeric_value"
        rows.append(row)
    return rows


def validate_export(
    manifest_csv,
    labels_dir,
    output_dir,
    images_dir=None,
    allowed_class_ids=None,
    require_label_file=False,
):
    manifest_csv = Path(manifest_csv)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if images_dir is not None:
        images_dir = Path(images_dir)

    manifest = pd.read_csv(manifest_csv)
    if "clean_image_path" not in manifest.columns:
        raise ValueError("Manifest must include clean_image_path.")

    if allowed_class_ids is None:
        allowed_class_ids = {0}
    else:
        allowed_class_ids = set(allowed_class_ids)

    manifest_stems = set()
    detail_rows = []
    label_rows = []

    for _, row in manifest.iterrows():
        manifest_image_path = Path(row["clean_image_path"])
        image_path = (
            images_dir / manifest_image_path.name
            if images_dir is not None
            else manifest_image_path
        )
        stem = image_path.stem
        manifest_stems.add(stem)
        label_path = labels_dir / f"{stem}.txt"
        image_exists = image_path.exists()
        image_width, image_height = read_image_size(image_path) if image_exists else (None, None)
        labels = parse_yolo_label(label_path)
        visible_box_count = 0
        errors = []

        if not image_exists:
            errors.append("missing_image")
        if require_label_file and not label_path.exists():
            errors.append("missing_label_file")

        for label in labels:
            label_error = label["parse_error"]
            if not label_error:
                if label["class_id"] not in allowed_class_ids:
                    label_error = "unexpected_class_id"
                elif label["width"] <= 0 or label["height"] <= 0:
                    label_error = "non_positive_box_size"
                elif label["width"] > 1 or label["height"] > 1:
                    label_error = "box_size_out_of_range"
                elif not (0 <= label["x_center"] <= 1 and 0 <= label["y_center"] <= 1):
                    label_error = "box_center_out_of_range"
                elif (
                    label["x_center"] - label["width"] / 2 < 0
                    or label["x_center"] + label["width"] / 2 > 1
                    or label["y_center"] - label["height"] / 2 < 0
                    or label["y_center"] + label["height"] / 2 > 1
                ):
                    label_error = "box_extends_outside_image"

            if label_error:
                errors.append(label_error)
            else:
                visible_box_count += 1

            label_rows.append({
                **{c: row.get(c) for c in ["review_order", "venue", "reason", "frame", "time_seconds"]},
                "image_stem": stem,
                "label_path": str(label_path),
                **label,
                "label_error": label_error,
            })

        if visible_box_count > 1:
            errors.append("multiple_ball_boxes")

        detail_rows.append({
            **row.to_dict(),
            "resolved_image_path": str(image_path),
            "label_path": str(label_path),
            "image_exists": image_exists,
            "image_width": image_width,
            "image_height": image_height,
            "label_file_exists": label_path.exists(),
            "label_line_count": len(labels),
            "valid_ball_box_count": visible_box_count,
            "is_null_frame": visible_box_count == 0,
            "validation_errors": ";".join(sorted(set(errors))),
        })

    detail = pd.DataFrame(detail_rows)
    labels = pd.DataFrame(label_rows)

    exported_label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    extra_label_stems = sorted(exported_label_stems - manifest_stems)
    missing_image_count = int((~detail["image_exists"]).sum())
    missing_label_count = int((~detail["label_file_exists"]).sum())
    invalid_rows = detail[detail["validation_errors"].fillna("") != ""].copy()

    detail_path = output_dir / "roboflow_ball_export_validation_details.csv"
    labels_path = output_dir / "roboflow_ball_export_validation_label_rows.csv"
    summary_path = output_dir / "roboflow_ball_export_validation_summary.csv"
    report_path = output_dir / "roboflow_ball_export_validation_report.md"
    detail.to_csv(detail_path, index=False)
    labels.to_csv(labels_path, index=False)

    summary_rows = []
    for cols in [["venue"], ["reason"], ["venue", "reason"]]:
        grouped = (
            detail.groupby(cols)
            .agg(
                frames=("review_order", "count"),
                visible_ball_frames=("is_null_frame", lambda s: int((~s).sum())),
                null_frames=("is_null_frame", lambda s: int(s.sum())),
                rows_with_errors=("validation_errors", lambda s: int((s.fillna("") != "").sum())),
            )
            .reset_index()
        )
        grouped.insert(0, "grouping", "+".join(cols))
        summary_rows.append(grouped)
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(summary_path, index=False)

    error_counts = (
        detail["validation_errors"]
        .fillna("")
        .str.split(";")
        .explode()
        .loc[lambda s: s != ""]
        .value_counts()
    )

    report_lines = [
        "# Roboflow Ball Export Validation",
        "",
        f"Manifest: `{manifest_csv}`",
        f"Labels dir: `{labels_dir}`",
        f"Images dir: `{images_dir or 'manifest paths'}`",
        "",
        f"Manifest rows: {len(detail)}",
        f"Images missing: {missing_image_count}",
        f"Label files missing: {missing_label_count}",
        f"Extra label files not in manifest: {len(extra_label_stems)}",
        f"Visible-ball frames: {int((~detail['is_null_frame']).sum())}",
        f"Null/no-ball frames: {int(detail['is_null_frame'].sum())}",
        f"Rows with validation errors: {len(invalid_rows)}",
        "",
        "Validation error counts:",
        "",
        error_counts.to_string() if len(error_counts) else "None",
        "",
        "Outputs:",
        "",
        f"- `{detail_path}`",
        f"- `{labels_path}`",
        f"- `{summary_path}`",
    ]
    if extra_label_stems:
        extra_path = output_dir / "roboflow_ball_export_extra_label_files.txt"
        extra_path.write_text("\n".join(extra_label_stems) + "\n")
        report_lines.append(f"- `{extra_path}`")

    report_path.write_text("\n".join(report_lines) + "\n")

    print("ROBOFLOW BALL EXPORT VALIDATION COMPLETE")
    print("----------------------------------------")
    print(f"Manifest rows: {len(detail)}")
    print(f"Visible-ball frames: {int((~detail['is_null_frame']).sum())}")
    print(f"Null/no-ball frames: {int(detail['is_null_frame'].sum())}")
    print(f"Rows with validation errors: {len(invalid_rows)}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a Roboflow YOLO export for the hard-frame ball batch."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional image directory. Defaults to clean_image_path values in manifest.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-class-ids", default="0")
    parser.add_argument(
        "--require-label-file",
        action="store_true",
        help="Flag missing .txt files as errors. By default missing labels count as null frames.",
    )
    args = parser.parse_args()

    validate_export(
        manifest_csv=args.manifest,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        allowed_class_ids=[
            int(value.strip())
            for value in args.allowed_class_ids.split(",")
            if value.strip()
        ],
        require_label_file=args.require_label_file,
    )


if __name__ == "__main__":
    main()
