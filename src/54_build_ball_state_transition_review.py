#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd


STATE_COLORS = {
    "controlled": (60, 220, 60),
    "in_transit": (0, 165, 255),
    "loose_or_unclear": (0, 0, 255),
    "unknown": (180, 180, 180),
}


def clean_id(value):
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.55):
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def load_video_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_number}")
    return frame


def annotate_frame(frame, row, title):
    state = row.get("ball_state", "unknown")
    color = STATE_COLORS.get(state, STATE_COLORS["unknown"])
    frame_number = int(row["frame"])
    time_seconds = float(row["time_seconds"])
    nearest_player = clean_id(row.get("nearest_player_id"))
    speed = row.get("ball_speed_px_per_second")
    distance = row.get("nearest_player_distance_px")
    status = row.get("association_status", "")
    source = row.get("source", "")

    if pd.notna(row.get("center_x")) and pd.notna(row.get("center_y")):
        bx = int(round(row["center_x"]))
        by = int(round(row["center_y"]))
        cv2.circle(frame, (bx, by), 9, color, -1)

        if all(pd.notna(row.get(c)) for c in ["x1", "y1", "x2", "y2"]):
            cv2.rectangle(
                frame,
                (int(round(row["x1"])), int(round(row["y1"]))),
                (int(round(row["x2"])), int(round(row["y2"]))),
                color,
                2,
            )

        if pd.notna(row.get("nearest_player_foot_x")) and pd.notna(
            row.get("nearest_player_foot_y")
        ):
            px = int(round(row["nearest_player_foot_x"]))
            py = int(round(row["nearest_player_foot_y"]))
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
            cv2.line(frame, (bx, by), (px, py), (255, 255, 255), 1)
            draw_label(frame, f"nearest: {nearest_player}", (px + 8, py), (255, 255, 255), 0.45)

    panel_lines = [
        title,
        f"Frame {frame_number}  Time {time_seconds:.2f}s",
        f"State: {state}",
        f"Nearest player: {nearest_player or '-'}",
        f"Status: {status or '-'}",
    ]
    if source:
        panel_lines.append(f"Source: {source}")
    if pd.notna(distance):
        panel_lines.append(f"Distance: {float(distance):.0f}px")
    if pd.notna(speed):
        panel_lines.append(f"Speed: {float(speed):.0f}px/s")

    panel_height = 26 + len(panel_lines) * 24
    cv2.rectangle(frame, (14, 14), (560, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (14, 14), (560, panel_height), color, 2)
    for index, line in enumerate(panel_lines):
        draw_label(frame, line, (26, 42 + index * 24), color if index == 2 else (255, 255, 255))

    return frame


def build_transition_review(
    video_path,
    association_csv,
    output_dir,
    start_time=None,
    end_time=None,
    max_pairs=None,
):
    video_path = Path(video_path)
    association_csv = Path(association_csv)
    output_dir = Path(output_dir)

    df = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    if "time_seconds" not in df.columns:
        raise ValueError("Association CSV must contain time_seconds")

    if start_time is not None:
        df = df[df["time_seconds"] >= start_time]
    if end_time is not None:
        df = df[df["time_seconds"] <= end_time]
    df = df.reset_index(drop=True)

    transitions = []
    for index in range(1, len(df)):
        before = df.iloc[index - 1]
        after = df.iloc[index]
        before_state = before.get("ball_state")
        after_state = after.get("ball_state")
        if pd.isna(before_state) or pd.isna(after_state):
            continue
        if before_state == after_state:
            continue
        transitions.append((before, after))
        if max_pairs is not None and len(transitions) >= max_pairs:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    manifest_rows = []
    for pair_index, (before, after) in enumerate(transitions, start=1):
        before_frame_number = int(before["frame"])
        after_frame_number = int(after["frame"])
        before_state = before["ball_state"]
        after_state = after["ball_state"]

        before_frame = load_video_frame(cap, before_frame_number)
        after_frame = load_video_frame(cap, after_frame_number)
        before_frame = annotate_frame(before_frame, before, "Frame 1: before switch")
        after_frame = annotate_frame(after_frame, after, "Frame 2: switch frame")

        combined = cv2.hconcat([before_frame, after_frame])
        safe_transition = f"{before_state}_to_{after_state}".replace("/", "-")
        filename = (
            f"{pair_index:04d}_{safe_transition}"
            f"_f{before_frame_number}_f{after_frame_number}"
            f"_t{float(after['time_seconds']):07.2f}.jpg"
        )
        output_path = pairs_dir / filename
        cv2.imwrite(str(output_path), combined)

        manifest_rows.append({
            "review_id": f"transition_{pair_index:04d}",
            "image_path": str(output_path),
            "before_frame": before_frame_number,
            "switch_frame": after_frame_number,
            "before_time": float(before["time_seconds"]),
            "switch_time": float(after["time_seconds"]),
            "before_state": before_state,
            "switch_state": after_state,
            "before_speed_px_per_second": before.get("ball_speed_px_per_second"),
            "switch_speed_px_per_second": after.get("ball_speed_px_per_second"),
            "before_distance_px": before.get("nearest_player_distance_px"),
            "switch_distance_px": after.get("nearest_player_distance_px"),
            "before_nearest_player_id": before.get("nearest_player_id"),
            "switch_nearest_player_id": after.get("nearest_player_id"),
            "before_association_status": before.get("association_status"),
            "switch_association_status": after.get("association_status"),
            "review_correct": "",
            "review_notes": "",
        })

    cap.release()

    manifest_path = output_dir / "transition_review_manifest.csv"
    columns = [
        "review_id",
        "image_path",
        "before_frame",
        "switch_frame",
        "before_time",
        "switch_time",
        "before_state",
        "switch_state",
        "before_speed_px_per_second",
        "switch_speed_px_per_second",
        "before_distance_px",
        "switch_distance_px",
        "before_nearest_player_id",
        "switch_nearest_player_id",
        "before_association_status",
        "switch_association_status",
        "review_correct",
        "review_notes",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("BALL STATE TRANSITION REVIEW COMPLETE")
    print("-------------------------------------")
    print(f"Video: {video_path}")
    print(f"Association CSV: {association_csv}")
    print(f"Transitions exported: {len(manifest_rows)}")
    print(f"Images: {pairs_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export side-by-side frames for reviewing ball-state transitions."
    )
    parser.add_argument("video")
    parser.add_argument("association_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    build_transition_review(
        video_path=args.video,
        association_csv=args.association_csv,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
