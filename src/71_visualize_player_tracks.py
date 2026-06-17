#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


def color_for_track(track_id):
    value = int(track_id) * 2654435761 % 255
    return (
        int((value * 3) % 255),
        int((value * 7 + 80) % 255),
        int((value * 11 + 140) % 255),
    )


def draw_label(frame, text, origin, color):
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def visualize_tracks(video_path, tracks_csv, output_path, start_time=0.0, end_time=None, min_track_age=1):
    video_path = Path(video_path)
    tracks_csv = Path(tracks_csv)
    output_path = Path(output_path)

    tracks = pd.read_csv(tracks_csv)
    tracks["frame"] = tracks["frame"].astype(int)
    if "track_id" not in tracks.columns:
        raise ValueError("Tracked player CSV must contain track_id.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(start_time * fps)))
    end_frame = total_frames if end_time is None else min(total_frames, int(round(end_time * fps)))

    rows_by_frame = {
        frame: group.to_dict("records")
        for frame, group in tracks[
            (tracks["frame"] >= start_frame) & (tracks["frame"] < end_frame)
        ].groupby("frame")
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_number = start_frame
    while frame_number < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rows = rows_by_frame.get(frame_number, [])
        for row in frame_rows:
            if int(row.get("track_age_frames", 1)) < min_track_age:
                continue
            track_id = int(row["track_id"])
            color = color_for_track(track_id)
            x1, y1, x2, y2 = [int(round(float(row[c]))) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"T{track_id}"
            if "detector_player_id" in row:
                label += f" d{row['detector_player_id']}"
            draw_label(frame, label, (x1, max(20, y1 - 7)), color)

        draw_label(
            frame,
            f"Frame {frame_number}  Time {frame_number / fps:.2f}s",
            (20, 35),
            (255, 255, 255),
        )
        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print("PLAYER TRACK OVERLAY COMPLETE")
    print("-----------------------------")
    print(f"Video: {video_path}")
    print(f"Tracks: {tracks_csv}")
    print(f"Output: {output_path}")
    print(f"Frames: {start_frame}-{end_frame}")


def main():
    parser = argparse.ArgumentParser(description="Render a video overlay of persistent player track IDs.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--min-track-age", type=int, default=3)
    args = parser.parse_args()

    visualize_tracks(
        video_path=args.video,
        tracks_csv=args.tracks,
        output_path=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        min_track_age=args.min_track_age,
    )


if __name__ == "__main__":
    main()
