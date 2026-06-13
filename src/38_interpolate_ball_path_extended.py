import argparse
from pathlib import Path

import pandas as pd


def interpolate_ball_path_extended(
    input_csv,
    output_csv,
    fps=30,
    max_gap_seconds=1.0,
    max_speed_px_per_second=2500,
):
    frame_col = "frame"
    position_cols = ["x1", "y1", "x2", "y2", "center_x", "center_y"]

    max_gap_frames = int(fps * max_gap_seconds)

    df = pd.read_csv(input_csv)

    if "source" not in df.columns:
        df["source"] = "detected"

    df = df.sort_values(frame_col).reset_index(drop=True)

    output_rows = []
    skipped_long_gaps = 0
    skipped_fast_gaps = 0
    interpolated_gaps = 0

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

        if missing_frames > max_gap_frames:
            skipped_long_gaps += 1
            continue

        dx = float(nxt["center_x"]) - float(curr["center_x"])
        dy = float(nxt["center_y"]) - float(curr["center_y"])
        distance_px = (dx ** 2 + dy ** 2) ** 0.5
        gap_seconds = gap / fps
        speed_px_per_second = distance_px / gap_seconds

        if speed_px_per_second > max_speed_px_per_second:
            skipped_fast_gaps += 1
            continue

        interpolated_gaps += 1

        for step in range(1, gap):
            alpha = step / gap
            interp = curr.copy()

            interp[frame_col] = f1 + step

            for col in position_cols:
                start_val = float(curr[col])
                end_val = float(nxt[col])
                interp[col] = start_val + alpha * (end_val - start_val)

            interp["time_seconds"] = interp[frame_col] / fps
            interp["source"] = "interpolated_extended"
            interp["confidence"] = None
            interp["object_type"] = "ball_interpolated_extended"

            output_rows.append(interp)

    output_rows.append(df.iloc[-1].to_dict())

    out = pd.DataFrame(output_rows)
    out = out.sort_values(frame_col).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    gaps_after = out[frame_col].diff().dropna()
    remaining_gaps_1s = ((gaps_after > 1) & (gaps_after <= max_gap_frames)).sum()

    print("EXTENDED INTERPOLATION COMPLETE")
    print("-------------------------------")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(out)}")
    print(f"Added rows: {len(out) - len(df)}")
    print("")
    print(f"Max interpolation gap: {max_gap_frames} frames / {max_gap_seconds:.2f}s")
    print(f"Max speed allowed: {max_speed_px_per_second:.0f} px/s")
    print("")
    print(f"Interpolated gaps: {interpolated_gaps}")
    print(f"Skipped long gaps: {skipped_long_gaps}")
    print(f"Skipped fast gaps: {skipped_fast_gaps}")
    print(f"Remaining gaps shorter than {max_gap_seconds:.2f}s: {remaining_gaps_1s}")


def main():
    parser = argparse.ArgumentParser(
        description="Cautiously interpolate ball trajectory gaps up to 1.0s with a speed gate."
    )

    parser.add_argument(
        "input_csv",
        help="Path to ball path CSV. Usually the 0.5s interpolated CSV."
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
        default=1.0,
        help="Maximum gap duration to interpolate. Default: 1.0"
    )

    parser.add_argument(
        "--max-speed",
        type=float,
        default=2500,
        help="Maximum allowed ball speed in px/s across gap. Default: 2500"
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)

    if args.output is None:
        output_path = input_path.with_name(
            input_path.stem.replace("_interpolated_ball_path", "")
            + "_extended_interpolated_ball_path.csv"
        )
    else:
        output_path = Path(args.output)

    interpolate_ball_path_extended(
        input_csv=input_path,
        output_csv=output_path,
        fps=args.fps,
        max_gap_seconds=args.max_gap_seconds,
        max_speed_px_per_second=args.max_speed,
    )


if __name__ == "__main__":
    main()