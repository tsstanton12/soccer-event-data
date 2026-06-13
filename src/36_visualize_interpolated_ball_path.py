import argparse
from pathlib import Path

import cv2
import pandas as pd


def visualize_ball_path(video_path, csv_path, output_path=None, trail_length=20):
    video_path = Path(video_path)
    csv_path = Path(csv_path)

    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_overlay.mp4")
    else:
        output_path = Path(output_path)

    df = pd.read_csv(csv_path)

    required_cols = ["frame", "x1", "y1", "x2", "y2", "center_x", "center_y", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["frame"] = df["frame"].astype(int)

    detections_by_frame = {
        frame: group.to_dict("records")
        for frame, group in df.groupby("frame")
    }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    recent_points = []

    frame_idx = 0

    print("VISUALIZING INTERPOLATED BALL PATH")
    print("----------------------------------")
    print(f"Video: {video_path}")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rows = detections_by_frame.get(frame_idx, [])

        for row in rows:
            source = row.get("source", "detected")

            x1 = int(round(row["x1"]))
            y1 = int(round(row["y1"]))
            x2 = int(round(row["x2"]))
            y2 = int(round(row["y2"]))
            cx = int(round(row["center_x"]))
            cy = int(round(row["center_y"]))

            if source == "interpolated":
                color = (0, 255, 255)  # yellow
                label = "interp"
            else:
                color = (0, 255, 0)  # green
                label = "detected"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            recent_points.append((cx, cy, source))

        recent_points = recent_points[-trail_length:]

        for i, (cx, cy, source) in enumerate(recent_points):
            if source == "interpolated":
                trail_color = (0, 255, 255)
            else:
                trail_color = (0, 255, 0)

            radius = max(2, int(5 * (i + 1) / len(recent_points)))
            cv2.circle(frame, (cx, cy), radius, trail_color, -1)

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_idx += 1

        if frame_idx % 1000 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print("DONE")
    print(f"Saved overlay video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize detected and interpolated ball tracking results."
    )

    parser.add_argument(
        "video_path",
        help="Path to original video file"
    )

    parser.add_argument(
        "csv_path",
        help="Path to interpolated ball path CSV"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output video path"
    )

    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of recent ball points to show as trail. Default: 20"
    )

    args = parser.parse_args()

    visualize_ball_path(
        video_path=args.video_path,
        csv_path=args.csv_path,
        output_path=args.output,
        trail_length=args.trail_length,
    )


if __name__ == "__main__":
    main()