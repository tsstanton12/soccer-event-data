# 41_interpolate_ball_path.py

import argparse
import pandas as pd
from pathlib import Path


def interpolate_ball_path(input_csv, output_csv, fps=30, max_gap_seconds=0.5):
    frame_col = "frame"
    frame_col = "frame"
    x_col = "center_x"
    y_col = "center_y"

    max_gap_frames = int(fps * max_gap_seconds)

    df = pd.read_csv(input_csv)

    if "source" not in df.columns:
        df["source"] = "detected"

    df = df.sort_values(frame_col).reset_index(drop=True)

    if "source" not in df.columns:
        df["source"] = "detected"

    output_rows = []

    for i in range(len(df) - 1):
        curr = df.iloc[i].to_dict()
        nxt = df.iloc[i + 1].to_dict()

        output_rows.append(curr)

        f1 = int(curr[frame_col])
        f2 = int(nxt[frame_col])
        gap = f2 - f1

        if gap <= 1:
            continue

        missing_frames = gap - 1

        if missing_frames <= max_gap_frames:
            position_cols = ["x1", "y1", "x2", "y2", "center_x", "center_y"]

            for step in range(1, gap):
                alpha = step / gap
                interp = curr.copy()

                interp[frame_col] = f1 + step

                for col in position_cols:
                    start_val = float(curr[col])
                    end_val = float(nxt[col])
                    interp[col] = start_val + alpha * (end_val - start_val)

                interp["time_seconds"] = interp[frame_col] / fps
                interp["source"] = "interpolated"
                interp["confidence"] = None
                interp["object_type"] = "ball_interpolated"

                output_rows.append(interp)

    output_rows.append(df.iloc[-1].to_dict())

    out = pd.DataFrame(output_rows)
    out = out.sort_values(frame_col).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    gaps_after = out[frame_col].diff().dropna()
    remaining_short_gaps = ((gaps_after > 1) & (gaps_after <= max_gap_frames)).sum()

    print("INTERPOLATION COMPLETE")
    print("----------------------")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(out)}")
    print(f"Detected rows: {(out['source'] == 'detected').sum()}")
    print(f"Interpolated rows: {(out['source'] == 'interpolated').sum()}")
    print(f"Max interpolation gap: {max_gap_frames} frames / {max_gap_seconds:.2f}s")
    print(f"Remaining gaps shorter than {max_gap_seconds:.2f}s: {remaining_short_gaps}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate short gaps in conservative ball tracking CSV."
    )

    parser.add_argument(
        "input_csv",
        help="Path to conservative ball path CSV"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output CSV path"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Video FPS. Default: 30"
    )

    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.5,
        help="Maximum gap duration to interpolate. Default: 0.5"
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)

    if args.output is None:
        output_path = input_path.with_name(
            input_path.stem.replace("_conservative_ball_path", "")
            + "_interpolated_ball_path.csv"
        )
    else:
        output_path = Path(args.output)

    interpolate_ball_path(
        input_csv=input_path,
        output_csv=output_path,
        fps=args.fps,
        max_gap_seconds=args.max_gap_seconds,
    )


if __name__ == "__main__":
    main()