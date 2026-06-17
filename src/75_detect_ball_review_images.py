#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def detect_review_images(model_path, image_dir, output_csv, confidence):
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    model = YOLO(str(model_path))
    rows = []
    for image_path in image_paths:
        result = model(str(image_path), conf=confidence, verbose=False)[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if str(label).lower() != "ball":
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "image_path": str(image_path),
                "filename": image_path.name,
                "object_type": "ball",
                "confidence": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("BALL REVIEW IMAGE DETECTION COMPLETE")
    print("------------------------------------")
    print(f"Model: {model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Detections: {len(rows)}")
    print(f"Output CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a YOLO ball model on review images and save predictions for evaluation."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    detect_review_images(
        model_path=args.model,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        confidence=args.conf,
    )


if __name__ == "__main__":
    main()
