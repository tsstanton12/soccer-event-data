#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


NON_ACTIVE_LABELS = {"referee", "substitute_or_staff", "off_field_player"}


def clean_id(value):
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_track_summary(path):
    summary = pd.read_csv(path)
    required = {
        "nearest_player_id",
        "reviewed_rows",
        "non_active_votes",
        "active_player_votes",
        "unclear_votes",
        "majority_binary",
        "majority_share",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Track summary missing required columns: {missing}")
    summary = summary.copy()
    summary["_track_id_clean"] = summary["nearest_player_id"].map(clean_id)
    return summary


def apply_track_eligibility_reviews(
    players_csv,
    track_summary_csv,
    output_csv,
    min_votes=1,
    min_majority_share=0.67,
):
    players_csv = Path(players_csv)
    track_summary_csv = Path(track_summary_csv)
    output_csv = Path(output_csv)

    players = pd.read_csv(players_csv)
    summary = load_track_summary(track_summary_csv)

    if "track_id" in players.columns:
        track_col = "track_id"
    elif "player_id" in players.columns:
        track_col = "player_id"
    else:
        raise ValueError("Player CSV must include track_id or player_id.")

    summary["exclude_track"] = (
        (summary["majority_binary"] == "non_active")
        & (summary["non_active_votes"] >= min_votes)
        & (summary["majority_share"] >= min_majority_share)
    )
    excluded_summary = summary[summary["exclude_track"]].copy()
    excluded_track_ids = set(excluded_summary["_track_id_clean"])

    players = players.copy()
    players["_track_id_clean"] = players[track_col].map(clean_id)
    removed = players[players["_track_id_clean"].isin(excluded_track_ids)].copy()
    filtered = players[~players["_track_id_clean"].isin(excluded_track_ids)].copy()

    filtered = filtered.drop(columns=["_track_id_clean"])
    removed = removed.drop(columns=["_track_id_clean"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    removed_path = output_csv.with_name(output_csv.stem + "_removed.csv")
    removed.to_csv(removed_path, index=False)
    excluded_summary_path = output_csv.with_name(output_csv.stem + "_excluded_track_summary.csv")
    excluded_summary.drop(columns=["_track_id_clean"], errors="ignore").to_csv(
        excluded_summary_path,
        index=False,
    )

    print("TRACK ELIGIBILITY FILTER COMPLETE")
    print("---------------------------------")
    print(f"Player rows: {len(players)}")
    print(f"Reviewed tracks: {len(summary)}")
    print(f"Excluded tracks: {len(excluded_track_ids)}")
    print(f"Removed player rows: {len(removed)}")
    print(f"Remaining player rows: {len(filtered)}")
    print(f"Filtered players: {output_csv}")
    print(f"Removed rows: {removed_path}")
    print(f"Excluded track summary: {excluded_summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove all detections belonging to tracks reviewed as non-active."
        )
    )
    parser.add_argument("players_csv")
    parser.add_argument("track_summary_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--min-majority-share", type=float, default=0.67)
    args = parser.parse_args()

    apply_track_eligibility_reviews(
        players_csv=args.players_csv,
        track_summary_csv=args.track_summary_csv,
        output_csv=args.output,
        min_votes=args.min_votes,
        min_majority_share=args.min_majority_share,
    )


if __name__ == "__main__":
    main()
