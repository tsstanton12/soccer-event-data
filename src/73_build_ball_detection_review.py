#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


REVIEW_LABELS = "correct, missed_ball, wrong_ball, poor_box, interpolation_bad, unsure"


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.55):
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def infer_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not infer FPS for video: {video_path}")
    return fps, total_frames


def add_motion_columns(ball, fps):
    ball = ball.sort_values("frame").drop_duplicates("frame", keep="first").reset_index(drop=True)
    ball["frame"] = ball["frame"].astype(int)
    ball["review_time_seconds"] = ball["frame"] / fps
    ball["prev_frame"] = ball["frame"].shift(1)
    ball["prev_center_x"] = ball["center_x"].shift(1)
    ball["prev_center_y"] = ball["center_y"].shift(1)
    ball["frame_gap"] = ball["frame"] - ball["prev_frame"]
    ball["time_gap_seconds"] = ball["frame_gap"] / fps
    ball["dx"] = ball["center_x"] - ball["prev_center_x"]
    ball["dy"] = ball["center_y"] - ball["prev_center_y"]
    ball["jump_px"] = ((ball["dx"] ** 2 + ball["dy"] ** 2) ** 0.5)
    ball["speed_px_per_second"] = ball["jump_px"] / ball["time_gap_seconds"]
    return ball


def choose_review_rows(ball, fps, max_items):
    per_reason = max(4, max_items // 4)
    candidates_by_reason = []

    low_conf = ball[
        ball["confidence"].notna()
        & (ball["confidence"] >= 0.15)
        & (ball["confidence"] <= 0.45)
    ].copy()
    candidates_by_reason.extend(
        ("low_confidence_detection", row)
        for _, row in low_conf.nsmallest(per_reason, "confidence").iterrows()
    )

    gaps = ball[ball["time_gap_seconds"] >= 0.50].copy()
    for _, row in gaps.nlargest(per_reason, "time_gap_seconds").iterrows():
        candidates_by_reason.append(("detection_gap_end", row))
        before = row.copy()
        before["frame"] = max(0, int(row["frame"] - round(row["frame_gap"] / 2)))
        before["review_time_seconds"] = before["frame"] / fps
        candidates_by_reason.append(("detection_gap_middle", before))

    jumps = ball[
        ball["speed_px_per_second"].notna()
        & (ball["time_gap_seconds"] <= 0.20)
        & (ball["speed_px_per_second"] >= 650)
    ].copy()
    candidates_by_reason.extend(
        ("suspicious_ball_jump", row)
        for _, row in jumps.nlargest(per_reason, "speed_px_per_second").iterrows()
    )

    if "source" in ball.columns:
        interpolated = ball[ball["source"].astype(str).str.contains("interpolated", na=False)].copy()
        for _, row in interpolated.iloc[:: max(1, len(interpolated) // max(1, per_reason))].head(per_reason).iterrows():
            candidates_by_reason.append(("interpolated_position", row))

    # Deduplicate by frame but keep the first reason, which is usually the most
    # specific because candidates are added in priority order.
    rows = []
    seen = set()
    for reason, row in candidates_by_reason:
        frame = int(row["frame"])
        if frame in seen:
            continue
        seen.add(frame)
        rows.append((reason, row))
        if len(rows) >= max_items:
            break

    if len(rows) < max_items:
        already = {int(row["frame"]) for _, row in rows}
        remaining = low_conf[~low_conf["frame"].astype(int).isin(already)]
        for _, row in remaining.nsmallest(max_items - len(rows), "confidence").iterrows():
            rows.append(("low_confidence_detection", row))
    return rows


def annotate_frame(frame, row, venue, reason):
    annotated = frame.copy()
    color = (0, 255, 255)
    if reason == "suspicious_ball_jump":
        color = (0, 0, 255)
    elif reason.startswith("detection_gap"):
        color = (255, 0, 255)
    elif reason == "low_confidence_detection":
        color = (0, 165, 255)

    if all(pd.notna(row.get(c)) for c in ["x1", "y1", "x2", "y2"]):
        x1, y1, x2, y2 = [int(round(float(row[c]))) for c in ["x1", "y1", "x2", "y2"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    if pd.notna(row.get("center_x")) and pd.notna(row.get("center_y")):
        cx = int(round(float(row["center_x"])))
        cy = int(round(float(row["center_y"])))
        cv2.circle(annotated, (cx, cy), 8, color, -1)

    lines = [
        f"{venue} / {reason}",
        f"frame {int(row['frame'])}  time {float(row.get('review_time_seconds', 0)):.2f}s",
        f"confidence {row.get('confidence', '')}",
        f"gap {float(row.get('time_gap_seconds', 0) or 0):.2f}s  speed {float(row.get('speed_px_per_second', 0) or 0):.0f}px/s",
        f"source {row.get('source', 'detected')}",
    ]
    cv2.rectangle(annotated, (12, 12), (720, 150), (0, 0, 0), -1)
    cv2.rectangle(annotated, (12, 12), (720, 150), color, 2)
    for index, line in enumerate(lines):
        draw_label(annotated, line, (26, 40 + index * 24), color if index == 0 else (255, 255, 255), 0.50)
    return annotated


def build_for_source(source, output_dir, order_start, max_items_per_source):
    venue, video, ball_csv = source
    video_path = Path(video)
    ball_path = Path(ball_csv)
    fps, _total_frames = infer_fps(video_path)
    ball = pd.read_csv(ball_path)
    required = ["frame", "x1", "y1", "x2", "y2", "center_x", "center_y"]
    missing = [column for column in required if column not in ball.columns]
    if missing:
        raise ValueError(f"{ball_path} missing required columns: {missing}")
    ball = add_motion_columns(ball, fps)
    review_rows = choose_review_rows(ball, fps, max_items_per_source)

    clean_dir = output_dir / "clean_frames"
    annotated_dir = output_dir / "annotated_frames"
    clean_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    manifest_rows = []
    order = order_start
    for reason, row in review_rows:
        frame_number = int(row["frame"])
        frame = read_frame(cap, frame_number)
        if frame is None:
            continue
        order += 1
        stem = f"{order:04d}_{venue}_{reason}_f{frame_number:06d}_t{float(row['review_time_seconds']):07.2f}"
        clean_path = clean_dir / f"{stem}.jpg"
        annotated_path = annotated_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(clean_path), frame)
        cv2.imwrite(str(annotated_path), annotate_frame(frame, row, venue, reason))
        manifest_rows.append({
            "review_order": order,
            "venue": venue,
            "video": str(video_path),
            "ball_csv": str(ball_path),
            "reason": reason,
            "frame": frame_number,
            "time_seconds": float(row["review_time_seconds"]),
            "confidence": row.get("confidence", ""),
            "source": row.get("source", "detected"),
            "time_gap_seconds": row.get("time_gap_seconds", ""),
            "speed_px_per_second": row.get("speed_px_per_second", ""),
            "x1": row.get("x1", ""),
            "y1": row.get("y1", ""),
            "x2": row.get("x2", ""),
            "y2": row.get("y2", ""),
            "center_x": row.get("center_x", ""),
            "center_y": row.get("center_y", ""),
            "clean_image_path": str(clean_path),
            "annotated_image_path": str(annotated_path),
            "review_label": "",
            "corrected_ball_visible": "",
            "review_notes": "",
        })
    cap.release()
    return manifest_rows, order


def parse_source(value):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--source must use NAME:VIDEO:BALL_CSV")
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-venue ball detection/tracking error review frames."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source spec as venue:video_path:ball_csv. Repeat for multiple venues.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items-per-source", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    order = 0
    for source_arg in args.source:
        rows, order = build_for_source(
            parse_source(source_arg),
            output_dir,
            order_start=order,
            max_items_per_source=args.max_items_per_source,
        )
        manifest_rows.extend(rows)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "ball_detection_review_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary_path = output_dir / "ball_detection_review_summary.md"
    summary_lines = [
        "# Ball Detection Review",
        "",
        f"Manifest: `{manifest_path}`",
        f"Clean frames: `{output_dir / 'clean_frames'}`",
        f"Annotated frames: `{output_dir / 'annotated_frames'}`",
        "",
        f"Recommended labels: `{REVIEW_LABELS}`",
        "",
        "Counts by venue/reason:",
        "",
        manifest.groupby(["venue", "reason"]).size().to_string() if len(manifest) else "No rows.",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines))

    print("BALL DETECTION REVIEW READY")
    print("---------------------------")
    print(f"Review frames: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"Clean frames: {output_dir / 'clean_frames'}")
    print(f"Annotated frames: {output_dir / 'annotated_frames'}")
    if len(manifest):
        print("")
        print(manifest.groupby(["venue", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
