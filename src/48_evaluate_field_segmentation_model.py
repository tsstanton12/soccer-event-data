#!/usr/bin/env python3

import argparse
import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def load_field_mask_module():
    path = Path(__file__).with_name("45_auto_field_mask.py")
    spec = importlib.util.spec_from_file_location("auto_field_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_videos(inputs):
    videos = []
    for value in inputs:
        path = Path(value)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
        elif path.is_dir():
            videos.extend(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(set(videos))


def mask_iou(first, second):
    intersection = np.count_nonzero((first > 0) & (second > 0))
    union = np.count_nonzero((first > 0) | (second > 0))
    return intersection / union if union else 0.0


def evaluate_video(video_path, model, field_mask, output_dir, sample_every, max_samples, conf, imgsz):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return []

    video_dir = output_dir / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    previous_mask = None
    frame_number = 0

    while len(rows) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            break

        poly, mask, area, confidence = field_mask.build_segmented_field_polygon(
            frame,
            model,
            max_points=6,
            conf=conf,
            imgsz=imgsz,
            inset_px=3,
        )
        h, w = frame.shape[:2]
        area_fraction = area / (h * w)
        temporal_iou = mask_iou(previous_mask, mask) if previous_mask is not None else ""
        status = "field_detected" if poly is not None else "no_field_detected"

        if poly is not None:
            previous_mask = mask

        debug_path = video_dir / f"frame_{frame_number:07d}.jpg"
        field_mask.save_debug(frame, poly, mask, debug_path)
        rows.append({
            "video": str(video_path),
            "frame": frame_number,
            "time_sec": round(frame_number / fps, 3) if fps else "",
            "status": status,
            "confidence": round(confidence, 4),
            "area_fraction": round(area_fraction, 4),
            "num_points": len(poly) if poly is not None else 0,
            "temporal_iou": round(temporal_iou, 4) if temporal_iou != "" else "",
            "debug_image": str(debug_path),
            "review": "",
        })
        frame_number += sample_every

    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/field_segmentation_evaluation")
    parser.add_argument("--sample_every", type=int, default=300)
    parser.add_argument("--max_samples_per_video", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(args.inputs)
    model = YOLO(args.model)
    field_mask = load_field_mask_module()

    rows = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video}")
        rows.extend(evaluate_video(
            video,
            model,
            field_mask,
            output_dir,
            args.sample_every,
            args.max_samples_per_video,
            args.conf,
            args.imgsz,
        ))

    report_path = output_dir / "evaluation_manifest.csv"
    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["video"])
        writer.writeheader()
        writer.writerows(rows)

    detected = sum(row["status"] == "field_detected" for row in rows)
    low_confidence = sum(row["confidence"] < 0.50 for row in rows)
    unstable = sum(row["temporal_iou"] != "" and row["temporal_iou"] < 0.70 for row in rows)
    print("\nFIELD SEGMENTATION EVALUATION")
    print("-----------------------------")
    print(f"Videos: {len(videos)}")
    print(f"Sampled frames: {len(rows)}")
    print(f"Fields detected: {detected}")
    print(f"Low-confidence frames: {low_confidence}")
    print(f"Large frame-to-frame changes: {unstable}")
    print(f"Review manifest: {report_path}")
    print("Set the manifest's review column to good, too_wide, too_tight, or missed.")


if __name__ == "__main__":
    main()
