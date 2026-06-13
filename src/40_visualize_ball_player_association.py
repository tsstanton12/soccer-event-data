import argparse
from pathlib import Path

import cv2
import pandas as pd


def visualize_association(video_path, association_csv, output_path=None):
    video_path = Path(video_path)
    association_csv = Path(association_csv)

    if output_path is None:
        output_path = association_csv.with_name(
            association_csv.stem + "_overlay.mp4"
        )
    else:
        output_path = Path(output_path)

    df = pd.read_csv(association_csv)
    df["frame"] = df["frame"].astype(int)

    required_cols = [
        "frame",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "nearest_player_id",
        "nearest_player_distance_px",
        "nearest_player_x",
        "nearest_player_y",
        "association_status",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Association CSV missing required columns: {missing}")

    rows_by_frame = {
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

    print("VISUALIZING BALL-PLAYER ASSOCIATION")
    print("-----------------------------------")
    print(f"Video: {video_path}")
    print(f"Association CSV: {association_csv}")
    print(f"Output: {output_path}")
    print(f"Frames: {total_frames}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rows = rows_by_frame.get(frame_idx, [])

        for row in rows:
            bx = int(round(row["center_x"]))
            by = int(round(row["center_y"]))

            ball_x1 = int(round(row["x1"]))
            ball_y1 = int(round(row["y1"]))
            ball_x2 = int(round(row["x2"]))
            ball_y2 = int(round(row["y2"]))

            source = row.get("source", "detected")
            status = row.get("association_status", "")

            if source in ["interpolated", "interpolated_extended"]:
                ball_color = (0, 255, 255)  # yellow
            else:
                ball_color = (0, 255, 0)  # green

            if status == "associated":
                line_color = (255, 0, 0)  # blue
            else:
                line_color = (0, 0, 255)  # red

            # Draw ball
            cv2.rectangle(
                frame,
                (ball_x1, ball_y1),
                (ball_x2, ball_y2),
                ball_color,
                2,
            )
            cv2.circle(frame, (bx, by), 6, ball_color, -1)

            # Draw nearest player point and line
            if pd.notna(row["nearest_player_x"]) and pd.notna(row["nearest_player_y"]):
                px = int(round(row["nearest_player_x"]))
                py = int(round(row["nearest_player_y"]))

                cv2.circle(frame, (px, py), 8, line_color, -1)
                cv2.line(frame, (bx, by), (px, py), line_color, 2)

                distance = row["nearest_player_distance_px"]
                player_id = row["nearest_player_id"]

                label = f"P{int(player_id)} {distance:.0f}px {status}"

                cv2.putText(
                    frame,
                    label,
                    (min(bx, px), max(25, min(by, py) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

            # Ball source label
            cv2.putText(
                frame,
                source,
                (ball_x1, max(20, ball_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ball_color,
                2,
                cv2.LINE_AA,
            )

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
        description="Visualize ball-player nearest-player association."
    )

    parser.add_argument("video_path", help="Path to original video")
    parser.add_argument("association_csv", help="Ball-player association CSV")

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output video path",
    )

    args = parser.parse_args()

    visualize_association(
        video_path=args.video_path,
        association_csv=args.association_csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()