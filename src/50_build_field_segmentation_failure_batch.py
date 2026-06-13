#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--neighbor_offsets", nargs="*", type=int, default=[-60, 0, 60])
    parser.add_argument("--include_reviews", nargs="+", default=["too_tight", "too_wide", "missed"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images_to_annotate"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.reviewed_manifest).open()))
    failures = [row for row in rows if row.get("review", "").strip() in args.include_reviews]
    output_rows = []

    for row in failures:
        video_path = Path(row["video"])
        source_frame = int(row["frame"])
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for offset in args.neighbor_offsets:
            frame_number = max(0, min(total_frames - 1, source_frame + offset))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue
            filename = f"{video_path.stem}_frame_{frame_number:07d}.jpg"
            image_path = images_dir / filename
            cv2.imwrite(str(image_path), frame)
            output_rows.append({
                "image": str(image_path),
                "video": str(video_path),
                "frame": frame_number,
                "time_sec": round(frame_number / fps, 3) if fps else "",
                "source_failure_frame": source_frame,
                "source_review": row["review"],
                "annotation_class": "playable_field",
                "annotation_status": "needs_polygon",
            })
        cap.release()

    unique_rows = {row["image"]: row for row in output_rows}
    manifest_path = output_dir / "annotation_manifest.csv"
    with manifest_path.open("w", newline="") as file:
        fieldnames = next(iter(unique_rows.values())).keys() if unique_rows else ["image"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows.values())

    print(f"Reviewed failures: {len(failures)}")
    print(f"Unique annotation frames: {len(unique_rows)}")
    print(f"Images: {images_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
