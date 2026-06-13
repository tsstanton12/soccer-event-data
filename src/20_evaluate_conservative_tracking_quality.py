import pandas as pd
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, help="Venue/output name, e.g. siena or patriot_league/lehigh")
    parser.add_argument("--ball-csv", required=True, help="Conservative ball path CSV from script 18")
    parser.add_argument("--gap-threshold", type=float, default=2.0, help="Large gap threshold in seconds")
    parser.add_argument("--output-report", default=None, help="Optional report text file path")
    args = parser.parse_args()

    ball_csv = Path(args.ball_csv)
    if not ball_csv.exists():
        raise FileNotFoundError(f"Ball CSV not found: {ball_csv}")

    ball = pd.read_csv(ball_csv)

    if len(ball) == 0:
        report = (
            "\nCONSERVATIVE BALL TRACKING REPORT\n"
            "---------------------------------\n"
            "No conservative ball detections found.\n"
        )
        print(report)
        return

    ball = ball.sort_values("frame").copy()

    video_duration = ball["time_seconds"].max() - ball["time_seconds"].min()

    ball["frame_gap"] = ball["frame"].diff()
    ball["time_gap"] = ball["time_seconds"].diff()

    total_detections = len(ball)
    frames_with_ball = ball["frame"].nunique()
    detections_per_second = total_detections / video_duration if video_duration > 0 else 0

    avg_frame_gap = ball["frame_gap"].mean()
    max_frame_gap = ball["frame_gap"].max()

    avg_time_gap = ball["time_gap"].mean()
    max_time_gap = ball["time_gap"].max()

    large_gaps = ball[ball["time_gap"] >= args.gap_threshold].copy()

    gaps_05 = len(ball[ball["time_gap"] >= 0.5])
    gaps_10 = len(ball[ball["time_gap"] >= 1.0])
    gaps_20 = len(ball[ball["time_gap"] >= 2.0])
    gaps_50 = len(ball[ball["time_gap"] >= 5.0])

    lines = []

    lines.append("\nCONSERVATIVE BALL TRACKING REPORT")
    lines.append("---------------------------------")
    lines.append(f"Venue: {args.venue}")
    lines.append(f"CSV: {ball_csv}")
    lines.append(f"Tracking duration covered: {video_duration:.1f} seconds")
    lines.append(f"Total conservative ball detections: {total_detections}")
    lines.append(f"Frames with ball detections: {frames_with_ball}")
    lines.append(f"Detections per second: {detections_per_second:.2f}")
    lines.append(f"Average frame gap: {avg_frame_gap:.1f}")
    lines.append(f"Max frame gap: {max_frame_gap:.1f}")
    lines.append(f"Average time gap: {avg_time_gap:.2f} seconds")
    lines.append(f"Max time gap: {max_time_gap:.2f} seconds")

    lines.append("\nGAP SUMMARY")
    lines.append("------------------")
    lines.append(f"Gaps >= 0.5 sec: {gaps_05}")
    lines.append(f"Gaps >= 1.0 sec: {gaps_10}")
    lines.append(f"Gaps >= 2.0 sec: {gaps_20}")
    lines.append(f"Gaps >= 5.0 sec: {gaps_50}")

    lines.append(f"\nLARGEST GAPS ({args.gap_threshold}+ sec)")
    lines.append("------------------")
    lines.append(f"Number of {args.gap_threshold}+ second gaps: {len(large_gaps)}")

    if len(large_gaps) > 0:
        gap_table = (
            large_gaps[[
                "frame",
                "time_seconds",
                "time_gap",
                "confidence",
                "center_x",
                "center_y",
            ]]
            .sort_values("time_gap", ascending=False)
            .head(15)
            .to_string(index=False)
        )
        lines.append(gap_table)

    lines.append("\nROUGH INTERPRETATION")
    lines.append("--------------------")

    if detections_per_second >= 2.0:
        lines.append("Detection density: GOOD")
    elif detections_per_second >= 0.75:
        lines.append("Detection density: USABLE")
    else:
        lines.append("Detection density: WEAK")

    if max_time_gap <= 2.0:
        lines.append("Continuity: GOOD")
    elif max_time_gap <= 5.0:
        lines.append("Continuity: USABLE WITH INTERPOLATION")
    else:
        lines.append("Continuity: GAPPY — useful in stretches, not full continuous tracking")

    report = "\n".join(lines)
    print(report)

    if args.output_report:
        output_report = Path(args.output_report)
    else:
        output_dir = Path("outputs") / args.venue.strip("/").replace("\\", "/")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_report = output_dir / f"{ball_csv.stem}_quality_report.txt"

    output_report.write_text(report)
    print(f"\nSaved report to {output_report}")


if __name__ == "__main__":
    main()