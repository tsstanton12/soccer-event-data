#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def frame_signature(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def discover_videos(paths):
    videos = []
    for value in paths:
        path = Path(value)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
        elif path.is_dir():
            videos.extend(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(set(videos))


def harvest_video(video_path, output_dir, samples_per_video, candidate_multiplier):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    candidate_count = max(samples_per_video, samples_per_video * candidate_multiplier)
    candidate_frames = np.linspace(0, max(0, total_frames - 1), candidate_count).astype(int)

    candidates = []
    previous_signature = None
    for frame_number in candidate_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = cap.read()
        if not ok:
            continue

        signature = frame_signature(frame)
        scene_change = 1.0 if previous_signature is None else cv2.compareHist(
            previous_signature,
            signature,
            cv2.HISTCMP_BHATTACHARYYA,
        )
        previous_signature = signature
        edge_density = float(np.mean(cv2.Canny(frame, 80, 180) > 0))
        candidates.append((scene_change + 0.25 * edge_density, int(frame_number), frame))

    cap.release()
    # Choose one strong candidate from each time segment. This prevents a few
    # dramatic camera changes from crowding out the rest of the match.
    selected = []
    segment_edges = np.linspace(0, len(candidates), samples_per_video + 1).astype(int)
    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        segment = candidates[start:end]
        if segment:
            selected.append(max(segment, key=lambda item: item[0]))
    selected.sort(key=lambda item: item[1])

    rows = []
    safe_stem = video_path.stem.replace(" ", "_")
    for _, frame_number, frame in selected:
        filename = f"{safe_stem}_frame_{frame_number:07d}.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)
        rows.append({
            "image": str(output_path),
            "video": str(video_path),
            "frame": frame_number,
            "time_sec": round(frame_number / fps, 3) if fps else "",
            "annotation_class": "playable_field",
            "annotation_status": "needs_polygon",
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data/field_segmentation/images_to_annotate")
    parser.add_argument("--samples_per_video", type=int, default=30)
    parser.add_argument("--candidate_multiplier", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(args.inputs)
    rows = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video}")
        rows.extend(harvest_video(video, output_dir, args.samples_per_video, args.candidate_multiplier))

    manifest_path = output_dir.parent / "annotation_manifest.csv"
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} frames from {len(videos)} videos")
    print(f"Images: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
