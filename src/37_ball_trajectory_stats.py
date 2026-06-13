import argparse
from pathlib import Path

import pandas as pd


def format_seconds(seconds):
    return f"{seconds:.2f}s"


def ball_trajectory_stats(csv_path, fps=30, output_csv=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = ["frame", "center_x", "center_y", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.sort_values("frame").reset_index(drop=True)
    df["frame"] = df["frame"].astype(int)

    total_rows = len(df)
    detected_rows = (df["source"] == "detected").sum()
    interpolated_rows = (df["source"] == "interpolated").sum()

    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    frame_span = last_frame - first_frame + 1
    coverage_pct = total_rows / frame_span * 100 if frame_span > 0 else 0

    df["frame_gap"] = df["frame"].diff()
    gaps = df[df["frame_gap"] > 1].copy()
    gaps["missing_frames"] = gaps["frame_gap"] - 1
    gaps["gap_seconds"] = gaps["missing_frames"] / fps

    gap_bins = {
        "1-15 frames": ((gaps["missing_frames"] >= 1) & (gaps["missing_frames"] <= 15)).sum(),
        "16-30 frames": ((gaps["missing_frames"] >= 16) & (gaps["missing_frames"] <= 30)).sum(),
        "31-60 frames": ((gaps["missing_frames"] >= 31) & (gaps["missing_frames"] <= 60)).sum(),
        "61-150 frames": ((gaps["missing_frames"] >= 61) & (gaps["missing_frames"] <= 150)).sum(),
        "151+ frames": (gaps["missing_frames"] >= 151).sum(),
    }

    # Continuous segment calculation
    segment_id = (df["frame_gap"].fillna(1) > 1).cumsum()
    segments = (
        df.groupby(segment_id)
        .agg(
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            rows=("frame", "count"),
        )
        .reset_index(drop=True)
    )
    segments["duration_seconds"] = (segments["end_frame"] - segments["start_frame"] + 1) / fps
    longest_segment = segments.sort_values("rows", ascending=False).iloc[0]

    # Speed sanity checks in pixels/second
    df["dx"] = df["center_x"].diff()
    df["dy"] = df["center_y"].diff()
    df["dt_frames"] = df["frame"].diff()
    df["dt_seconds"] = df["dt_frames"] / fps
    df["distance_px"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5
    df["speed_px_per_second"] = df["distance_px"] / df["dt_seconds"]

    valid_speeds = df[
        (df["dt_seconds"] > 0) &
        (df["dt_frames"] <= 15) &
        (df["speed_px_per_second"].notna())
    ]

    avg_speed = valid_speeds["speed_px_per_second"].mean()
    median_speed = valid_speeds["speed_px_per_second"].median()
    max_speed = valid_speeds["speed_px_per_second"].max()

    # Flag suspicious jumps
    suspicious_jumps = valid_speeds[
        valid_speeds["speed_px_per_second"] > 2500
    ][["frame", "frame_gap", "distance_px", "speed_px_per_second", "source"]]

    print("BALL TRAJECTORY STATS")
    print("---------------------")
    print(f"CSV: {csv_path}")
    print(f"FPS: {fps}")
    print("")
    print(f"Total rows: {total_rows}")
    print(f"Detected rows: {detected_rows}")
    print(f"Interpolated rows: {interpolated_rows}")
    print(f"Interpolated share: {interpolated_rows / total_rows * 100:.1f}%")
    print("")
    print(f"First frame: {first_frame}")
    print(f"Last frame: {last_frame}")
    print(f"Frame span: {frame_span}")
    print(f"Coverage: {coverage_pct:.1f}%")
    print("")
    print("Gap histogram:")
    for label, count in gap_bins.items():
        print(f"  {label}: {count}")
    print("")
    print(f"Total remaining gaps: {len(gaps)}")

    if len(gaps) > 0:
        largest_gap = gaps.sort_values("missing_frames", ascending=False).iloc[0]
        print(
            f"Largest remaining gap: {int(largest_gap['missing_frames'])} frames "
            f"/ {format_seconds(largest_gap['gap_seconds'])}"
        )
    else:
        print("Largest remaining gap: none")

    print("")
    print(f"Continuous segments: {len(segments)}")
    print(
        f"Longest segment: frames {int(longest_segment['start_frame'])}–"
        f"{int(longest_segment['end_frame'])} "
        f"({format_seconds(longest_segment['duration_seconds'])})"
    )
    print("")
    print("Speed sanity checks:")
    print(f"  Average speed: {avg_speed:.1f} px/s")
    print(f"  Median speed: {median_speed:.1f} px/s")
    print(f"  Max speed: {max_speed:.1f} px/s")
    print(f"  Suspicious jumps >2500 px/s: {len(suspicious_jumps)}")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        suspicious_jumps.to_csv(output_csv, index=False)
        print("")
        print(f"Suspicious jumps saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stats for interpolated ball trajectory CSV."
    )

    parser.add_argument(
        "csv_path",
        help="Path to interpolated ball path CSV"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Video FPS. Default: 30"
    )

    parser.add_argument(
        "--output-jumps",
        default=None,
        help="Optional CSV path for suspicious jump rows"
    )

    args = parser.parse_args()

    ball_trajectory_stats(
        csv_path=args.csv_path,
        fps=args.fps,
        output_csv=args.output_jumps,
    )


if __name__ == "__main__":
    main()