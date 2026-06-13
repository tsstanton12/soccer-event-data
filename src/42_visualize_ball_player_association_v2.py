import argparse
from pathlib import Path

import cv2
import pandas as pd


def visualize_association_v2(video_path, association_csv, output_path=None):
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

    print("VISUALIZING BALL-PLAYER ASSOCIATION V2")
    print("--------------------------------------")
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

            ball_state = row.get("ball_state", "unknown")
            status = row.get("association_status", "unknown")
            source = row.get("source", "detected")

            if ball_state == "controlled":
                color = (255, 0, 0)      # blue
            elif ball_state == "in_transit":
                color = (0, 165, 255)    # orange
            elif ball_state == "loose_or_unclear":
                color = (0, 0, 255)      # red
            else:
                color = (180, 180, 180)  # gray

            # Draw ball point
            cv2.circle(frame, (bx, by), 7, color, -1)

            # Draw ball box if available
            if all(pd.notna(row.get(c)) for c in ["x1", "y1", "x2", "y2"]):
                x1 = int(round(row["x1"]))
                y1 = int(round(row["y1"]))
                x2 = int(round(row["x2"]))
                y2 = int(round(row["y2"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw nearest player foot point and line
            if (
                pd.notna(row.get("nearest_player_foot_x"))
                and pd.notna(row.get("nearest_player_foot_y"))
            ):
                px = int(round(row["nearest_player_foot_x"]))
                py = int(round(row["nearest_player_foot_y"]))

                cv2.circle(frame, (px, py), 8, color, -1)

                if ball_state == "controlled":
                    cv2.line(frame, (bx, by), (px, py), color, 2)
                elif ball_state == "in_transit":
                    cv2.line(frame, (bx, by), (px, py), color, 1)
                else:
                    cv2.line(frame, (bx, by), (px, py), color, 1)

                distance = row.get("nearest_player_distance_px")
                speed = row.get("ball_speed_px_per_second")

                label = f"{ball_state}"

                if pd.notna(distance):
                    label += f" {distance:.0f}px"

                if pd.notna(speed):
                    label += f" {speed:.0f}px/s"

                cv2.putText(
                    frame,
                    label,
                    (min(bx, px), max(25, min(by, py) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Status label near ball
            cv2.putText(
                frame,
                f"{source} | {status}",
                (bx + 10, max(25, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
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
        description="Visualize V2 ball-player association with ball states."
    )

    parser.add_argument("video_path")
    parser.add_argument("association_csv")
    parser.add_argument("--output", "-o", default=None)

    args = parser.parse_args()

    visualize_association_v2(
        video_path=args.video_path,
        association_csv=args.association_csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()