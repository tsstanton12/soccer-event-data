import cv2
import pandas as pd
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--venue",
        required=True,
        help="Venue name, e.g. siena"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Video file path"
    )

    parser.add_argument(
        "--ball-csv",
        required=True,
        help="Conservative tracking CSV"
    )

    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=2.0
    )

    args = parser.parse_args()

    output_dir = (
        Path("outputs")
        / args.venue
        / "gap_review_frames"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    ball = pd.read_csv(args.ball_csv)
    ball = ball.sort_values("frame").copy()

    ball["time_gap"] = ball["time_seconds"].diff()

    large_gaps = ball[
        ball["time_gap"] >= args.gap_threshold
    ].copy()

    cap = cv2.VideoCapture(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    saved = 0

    print(
        f"Found {len(large_gaps)} gaps "
        f"of {args.gap_threshold}+ seconds."
    )

    print("Exporting review frames...")

    for gap_index, row in large_gaps.iterrows():

        gap_end_time = row["time_seconds"]
        gap_start_time = gap_end_time - row["time_gap"]

        export_start = max(
            0,
            gap_start_time - 2.0
        )

        export_end = gap_end_time + 2.0

        t = export_start

        while t <= export_end:

            frame_num = int(t * fps)

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_num
            )

            ret, frame = cap.read()

            if ret:

                output_path = (
                    output_dir /
                    f"gap_{int(gap_index):03d}"
                    f"_time_{t:07.2f}"
                    f"_frame_{frame_num:06d}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    frame
                )

                saved += 1

            t += 0.5

    cap.release()

    print(
        f"Done! Saved {saved} frames "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()