#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


NON_LIVE_LABELS = {"dead_ball", "restart_setup", "goal_stoppage"}


def normalize_text(value, default=""):
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def build_intervals(review_csv, include_live=False):
    reviews = pd.read_csv(review_csv)
    required = ["start_time", "end_time", "game_state_label"]
    missing = [column for column in required if column not in reviews.columns]
    if missing:
        raise ValueError(f"Missing required review columns: {missing}")

    intervals = []
    for _, row in reviews.iterrows():
        label = normalize_text(row.get("game_state_label"), "unknown")
        if label == "unknown":
            continue
        if not include_live and label == "live":
            continue
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        if end_time <= start_time:
            continue
        restart_type = normalize_text(row.get("restart_type"), "none")
        intervals.append({
            "start_time": start_time,
            "end_time": end_time,
            "game_state_label": label,
            "restart_type": restart_type,
            "is_live": label not in NON_LIVE_LABELS,
            "source_review_id": normalize_text(row.get("review_id")),
            "source_event_impact_label": normalize_text(row.get("source_event_impact_label")),
            "source_notes": normalize_text(row.get("source_notes")),
        })

    return pd.DataFrame(
        intervals,
        columns=[
            "start_time",
            "end_time",
            "game_state_label",
            "restart_type",
            "is_live",
            "source_review_id",
            "source_event_impact_label",
            "source_notes",
        ],
    )


def apply_intervals(frames_csv, review_csv, output_frames_csv, output_intervals_csv, include_live=False):
    frames_csv = Path(frames_csv)
    review_csv = Path(review_csv)
    output_frames_csv = Path(output_frames_csv)
    output_intervals_csv = Path(output_intervals_csv)

    frames = pd.read_csv(frames_csv)
    if "time_seconds" not in frames.columns:
        raise ValueError("Frame CSV must contain time_seconds")

    intervals = build_intervals(review_csv, include_live=include_live)

    frames["game_state_label"] = "live"
    frames["game_state_is_live"] = True
    frames["game_state_restart_type"] = "none"
    frames["game_state_source_review_id"] = ""
    frames["game_state_source_event_impact_label"] = ""

    for _, interval in intervals.iterrows():
        mask = (
            (frames["time_seconds"] >= float(interval["start_time"]))
            & (frames["time_seconds"] <= float(interval["end_time"]))
        )
        frames.loc[mask, "game_state_label"] = interval["game_state_label"]
        frames.loc[mask, "game_state_is_live"] = bool(interval["is_live"])
        frames.loc[mask, "game_state_restart_type"] = interval["restart_type"]
        frames.loc[mask, "game_state_source_review_id"] = interval["source_review_id"]
        frames.loc[mask, "game_state_source_event_impact_label"] = interval[
            "source_event_impact_label"
        ]

    output_frames_csv.parent.mkdir(parents=True, exist_ok=True)
    output_intervals_csv.parent.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output_frames_csv, index=False)
    intervals.to_csv(output_intervals_csv, index=False)

    summary = frames["game_state_label"].value_counts(dropna=False)
    non_live_rows = int((~frames["game_state_is_live"]).sum())

    print("GAME-STATE REVIEW APPLICATION COMPLETE")
    print("--------------------------------------")
    print(f"Input frames: {frames_csv}")
    print(f"Reviewed intervals: {len(intervals)}")
    print(f"Frame rows: {len(frames)}")
    print(f"Non-live frame rows: {non_live_rows}")
    print("")
    print(summary.to_string())
    print("")
    print(f"Output frames: {output_frames_csv}")
    print(f"Output intervals: {output_intervals_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply reviewed live/dead/restart windows to a frame-level possession CSV."
    )
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--output-intervals", required=True)
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Also write reviewed live windows as explicit intervals. By default only non-live/unknown windows are intervalized.",
    )
    args = parser.parse_args()

    apply_intervals(
        frames_csv=args.frames_csv,
        review_csv=args.review_csv,
        output_frames_csv=args.output_frames,
        output_intervals_csv=args.output_intervals,
        include_live=args.include_live,
    )


if __name__ == "__main__":
    main()
