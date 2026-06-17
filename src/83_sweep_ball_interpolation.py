#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from importlib import import_module


interpolate_module = import_module("38_interpolate_ball_path_extended")
association_module = import_module("41_ball_player_association_v2")
chains_module = import_module("52_build_possession_chains")


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_or_float_label(value):
    text = str(value).replace(".", "p")
    return text


def summarize_ball_path(csv_path, fps, suspicious_speed):
    df = pd.read_csv(csv_path).sort_values("frame").reset_index(drop=True)
    df["frame"] = df["frame"].astype(int)
    frame_span = int(df["frame"].max() - df["frame"].min() + 1) if len(df) else 0
    df["frame_gap"] = df["frame"].diff()
    df["missing_frames"] = df["frame_gap"] - 1
    gaps = df[df["frame_gap"] > 1].copy()

    df["dx"] = df["center_x"].diff()
    df["dy"] = df["center_y"].diff()
    df["dt_seconds"] = df["frame_gap"] / fps
    df["distance_px"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5
    df["speed_px_per_second"] = df["distance_px"] / df["dt_seconds"]
    valid_speed = df[(df["dt_seconds"] > 0) & df["speed_px_per_second"].notna()]
    suspicious = valid_speed[valid_speed["speed_px_per_second"] > suspicious_speed]

    source = df["source"].fillna("detected").astype(str) if "source" in df.columns else pd.Series(["detected"] * len(df))
    interpolated_rows = int(source.str.contains("interpolated").sum())

    return {
        "ball_rows": int(len(df)),
        "frame_span": frame_span,
        "coverage_rate": float(len(df) / frame_span) if frame_span else 0.0,
        "detected_rows": int((~source.str.contains("interpolated")).sum()),
        "interpolated_rows": interpolated_rows,
        "interpolated_share": float(interpolated_rows / len(df)) if len(df) else 0.0,
        "remaining_gap_count": int(len(gaps)),
        "max_remaining_gap_frames": int(gaps["missing_frames"].max()) if len(gaps) else 0,
        "remaining_gap_frames_total": int(gaps["missing_frames"].sum()) if len(gaps) else 0,
        "median_speed_px_per_second": float(valid_speed["speed_px_per_second"].median()) if len(valid_speed) else 0.0,
        "p95_speed_px_per_second": float(valid_speed["speed_px_per_second"].quantile(0.95)) if len(valid_speed) else 0.0,
        "observed_max_speed_px_per_second": float(valid_speed["speed_px_per_second"].max()) if len(valid_speed) else 0.0,
        "suspicious_speed_jumps": int(len(suspicious)),
    }


def summarize_chains(frames_csv, segments_csv):
    frames = pd.read_csv(frames_csv)
    segments = pd.read_csv(segments_csv)
    return {
        "controlled_frames": int((frames["possession_state"] == "controlled").sum())
        if "possession_state" in frames.columns
        else 0,
        "in_transit_frames": int((frames["possession_state"] == "in_transit").sum())
        if "possession_state" in frames.columns
        else 0,
        "loose_or_unclear_frames": int((frames["possession_state"] == "loose_or_unclear").sum())
        if "possession_state" in frames.columns
        else 0,
        "pending_control_frames": int((frames["possession_state"] == "pending_control").sum())
        if "possession_state" in frames.columns
        else 0,
        "possession_segments": int(len(segments)),
        "total_controlled_duration_seconds": float(segments["duration_seconds"].sum())
        if len(segments) and "duration_seconds" in segments.columns
        else 0.0,
        "median_segment_duration_seconds": float(segments["duration_seconds"].median())
        if len(segments) and "duration_seconds" in segments.columns
        else 0.0,
    }


def run_sweep(args):
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for max_gap_seconds in parse_float_list(args.max_gap_seconds):
        for max_speed in parse_float_list(args.max_speeds):
            label = (
                f"gap{parse_int_or_float_label(max_gap_seconds)}"
                f"_speed{int(max_speed)}"
            )
            run_dir = output_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            interpolated_csv = run_dir / f"{input_csv.stem}_{label}_ball_path.csv"

            interpolate_module.interpolate_ball_path_extended(
                input_csv=input_csv,
                output_csv=interpolated_csv,
                fps=args.fps,
                max_gap_seconds=max_gap_seconds,
                max_speed_px_per_second=max_speed,
            )
            row = {
                "label": label,
                "max_gap_seconds": max_gap_seconds,
                "max_speed_px_per_second": max_speed,
                "interpolated_csv": str(interpolated_csv),
                **summarize_ball_path(
                    interpolated_csv,
                    fps=args.fps,
                    suspicious_speed=args.suspicious_speed,
                ),
            }

            if args.players_csv:
                association_csv = run_dir / f"{input_csv.stem}_{label}_association.csv"
                frames_csv = run_dir / f"{input_csv.stem}_{label}_possession_frames.csv"
                segments_csv = run_dir / f"{input_csv.stem}_{label}_possession_segments.csv"
                association_module.associate_ball_to_on_field_players(
                    ball_csv=interpolated_csv,
                    player_csv=args.players_csv,
                    output_csv=association_csv,
                    fps=args.fps,
                    max_distance_px=args.max_distance_px,
                    control_distance_px=args.control_distance_px,
                    controlled_speed_px_per_second=args.controlled_speed,
                    transit_speed_px_per_second=args.transit_speed,
                    control_field_zones=[
                        zone.strip()
                        for zone in args.control_field_zones.split(",")
                        if zone.strip()
                    ],
                    recompute_time_seconds=True,
                )
                chains_module.build_possession_chains(
                    association_csv=association_csv,
                    output_frames_csv=frames_csv,
                    output_segments_csv=segments_csv,
                    fps=args.fps,
                    min_confirm_frames=args.min_confirm_frames,
                    max_control_distance_px=args.max_control_distance_px,
                    max_carry_forward_seconds=args.max_carry_forward_seconds,
                    track_stability_csv=args.track_stability_csv,
                    exclude_track_stability_flags=[
                        flag.strip()
                        for flag in args.exclude_track_stability_flags.split(",")
                        if flag.strip()
                    ],
                )
                row.update({
                    "association_csv": str(association_csv),
                    "frames_csv": str(frames_csv),
                    "segments_csv": str(segments_csv),
                    **summarize_chains(frames_csv, segments_csv),
                })

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "interpolation_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# Ball Interpolation Sweep",
        "",
        f"Input: `{input_csv}`",
        f"FPS: {args.fps}",
        f"Runs: {len(summary)}",
        "",
        "Lower remaining gaps is good, but high interpolated share and many",
        "suspicious speed jumps mean the path may be inventing too much motion.",
        "",
        "## Summary",
        "",
        summary[
            [
                "label",
                "coverage_rate",
                "interpolated_share",
                "remaining_gap_count",
                "max_remaining_gap_frames",
                "suspicious_speed_jumps",
            ]
            + (["possession_segments", "controlled_frames", "in_transit_frames"] if args.players_csv else [])
        ].to_string(index=False),
        "",
        f"CSV: `{summary_path}`",
    ]
    report_path = output_dir / "interpolation_sweep_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print("BALL INTERPOLATION SWEEP COMPLETE")
    print("---------------------------------")
    print(f"Runs: {len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep ball interpolation settings and optionally score possession impact."
    )
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--max-gap-seconds", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-speeds", default="1200,1800,2500,3500")
    parser.add_argument("--suspicious-speed", type=float, default=2500)
    parser.add_argument("--players-csv", default=None)
    parser.add_argument("--track-stability-csv", default=None)
    parser.add_argument("--max-distance-px", type=float, default=150)
    parser.add_argument("--control-distance-px", type=float, default=80)
    parser.add_argument("--controlled-speed", type=float, default=500)
    parser.add_argument("--transit-speed", type=float, default=700)
    parser.add_argument("--control-field-zones", default="strict,tolerant")
    parser.add_argument("--min-confirm-frames", type=int, default=15)
    parser.add_argument("--max-control-distance-px", type=float, default=80)
    parser.add_argument("--max-carry-forward-seconds", type=float, default=2.0)
    parser.add_argument(
        "--exclude-track-stability-flags",
        default="short_track,jumpy_track,gappy_track",
    )
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
