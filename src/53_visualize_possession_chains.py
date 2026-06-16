#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


STATE_COLORS = {
    "controlled": (60, 220, 60),
    "pending_control": (0, 215, 255),
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


def draw_label(frame, text, origin, color, scale=0.55):
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


def visualize_possession_chains(
    video_path,
    chain_frames_csv,
    output_path,
    max_frames=None,
):
    video_path = Path(video_path)
    chain_frames_csv = Path(chain_frames_csv)
    output_path = Path(output_path)

    df = pd.read_csv(chain_frames_csv)
    df["frame"] = df["frame"].astype(int)
    rows_by_frame = {
        int(frame): group.iloc[0].to_dict()
        for frame, group in df.groupby("frame")
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    print("VISUALIZING POSSESSION CHAINS")
    print("-----------------------------")
    print(f"Video: {video_path}")
    print(f"Chain frames: {chain_frames_csv}")
    print(f"Output: {output_path}")
    print(f"Frames to render: {total_frames}")

    frame_idx = 0
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        row = rows_by_frame.get(frame_idx)
        if row:
            bx = int(round(row["center_x"]))
            by = int(round(row["center_y"]))
            possession_state = row.get("possession_state", "unknown")
            raw_state = row.get("ball_state", "unknown")
            possession_player = clean_id(row.get("possession_player_id"))
            nearest_player = clean_id(row.get("nearest_player_id"))
            pending_player = clean_id(row.get("pending_player_id"))
            pending_count = int(row.get("pending_confirm_frames", 0) or 0)
            carry_forward_count = int(row.get("carry_forward_frames", 0) or 0)
            carry_forward_seconds = row.get("carry_forward_seconds", 0)
            segment_id = row.get("possession_segment_id", "")
            transition_reason = row.get("transition_reason", "")
            speed = row.get("ball_speed_px_per_second")
            distance = row.get("nearest_player_distance_px")

            # Color the ball by the raw physical state so green means the ball
            # is currently classified as controlled. The chain state remains in
            # the panel as ownership confidence/context.
            color = STATE_COLORS.get(raw_state, STATE_COLORS["unknown"])

            cv2.circle(frame, (bx, by), 9, color, -1)
            if all(pd.notna(row.get(c)) for c in ["x1", "y1", "x2", "y2"]):
                cv2.rectangle(
                    frame,
                    (int(row["x1"]), int(row["y1"])),
                    (int(row["x2"]), int(row["y2"])),
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
                draw_label(frame, f"raw nearest: {nearest_player}", (px + 8, py), (255, 255, 255), 0.45)

            panel_lines = [
                f"Frame {frame_idx}  Time {row.get('time_seconds', 0):.2f}s",
                f"Raw: {raw_state} nearest={nearest_player}",
                f"Chain: {possession_state} owner={possession_player or '-'}",
                f"Ball color: {raw_state}",
                f"Segment: {segment_id or '-'}",
            ]
            if pending_player:
                panel_lines.append(f"Pending: {pending_player} ({pending_count} frames)")
            if carry_forward_count:
                panel_lines.append(
                    f"Carry-forward: {carry_forward_count} frames / {float(carry_forward_seconds):.2f}s"
                )
            if pd.notna(distance):
                panel_lines.append(f"Distance: {float(distance):.0f}px")
            if pd.notna(speed):
                panel_lines.append(f"Speed: {float(speed):.0f}px/s")
            if transition_reason:
                panel_lines.append(f"Transition: {transition_reason}")

            panel_height = 26 + len(panel_lines) * 24
            cv2.rectangle(frame, (14, 14), (520, panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (14, 14), (520, panel_height), color, 2)
            for index, line in enumerate(panel_lines):
                draw_label(frame, line, (26, 42 + index * 24), color if index == 2 else (255, 255, 255))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"Rendered {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print("DONE")
    print(f"Saved overlay: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a review overlay for smoothed possession chains."
    )
    parser.add_argument("video")
    parser.add_argument("chain_frames_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    visualize_possession_chains(
        video_path=args.video,
        chain_frames_csv=args.chain_frames_csv,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
