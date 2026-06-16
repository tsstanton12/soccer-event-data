#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


NON_ACTIVE_LABELS = {"referee", "substitute_or_staff", "off_field_player"}
NON_ACTIVE_NOTE_PATTERN = re.compile(
    r"referee|substitute|subsitute|staff|off[-_ ]field", re.IGNORECASE
)
RANK_PATTERN = re.compile(r"#\s*(\d+)")


def clean_id(value):
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def apply_player_eligibility_reviews(
    player_csv,
    review_csv,
    output_csv,
    association_csv=None,
    nearest_limit=5,
    non_active_labels=None,
):
    player_csv = Path(player_csv)
    review_csv = Path(review_csv)
    output_csv = Path(output_csv)

    players = pd.read_csv(player_csv)
    reviews = pd.read_csv(review_csv)
    association = pd.read_csv(association_csv) if association_csv else None

    if non_active_labels is None:
        non_active_labels = NON_ACTIVE_LABELS
    else:
        non_active_labels = set(non_active_labels)

    required_review_cols = ["frame", "nearest_player_id", "eligibility_label"]
    missing = [c for c in required_review_cols if c not in reviews.columns]
    if missing:
        raise ValueError(f"Missing columns in review CSV: {missing}")

    players["_eligibility_key"] = (
        players["frame"].astype(int).astype(str)
        + ":"
        + players["player_id"].map(clean_id)
    )

    reviews["eligibility_label"] = reviews["eligibility_label"].fillna("").str.strip()
    reviews["_eligibility_key"] = (
        reviews["frame"].astype(int).astype(str)
        + ":"
        + reviews["nearest_player_id"].map(clean_id)
    )

    exclusions = reviews[reviews["eligibility_label"].isin(non_active_labels)].copy()
    exclusion_keys = set(exclusions["_eligibility_key"])

    note_exclusion_keys = set()
    if association is not None:
        required_association_cols = ["frame", "center_x", "center_y"]
        missing = [c for c in required_association_cols if c not in association.columns]
        if missing:
            raise ValueError(f"Missing columns in association CSV: {missing}")

        if "foot_x" not in players.columns:
            players["foot_x"] = (players["x1"] + players["x2"]) / 2
        if "foot_y" not in players.columns:
            players["foot_y"] = players["y2"]

        ball_by_frame = {
            int(row["frame"]): row
            for _, row in association[required_association_cols].iterrows()
        }

        for _, review in reviews.iterrows():
            notes = str(review.get("review_notes") or "")
            if not NON_ACTIVE_NOTE_PATTERN.search(notes):
                continue

            ranks = [int(match.group(1)) for match in RANK_PATTERN.finditer(notes)]
            if not ranks:
                continue

            frame = int(review["frame"])
            ball = ball_by_frame.get(frame)
            if ball is None:
                continue

            frame_players = players[players["frame"].astype(int) == frame].copy()
            if frame_players.empty:
                continue

            frame_players["_distance_to_ball_px"] = (
                (frame_players["foot_x"] - float(ball["center_x"])) ** 2
                + (frame_players["foot_y"] - float(ball["center_y"])) ** 2
            ) ** 0.5
            ranked = frame_players.sort_values("_distance_to_ball_px").head(nearest_limit)
            ranked = ranked.reset_index(drop=True)
            for rank in ranks:
                if rank < 1 or rank > len(ranked):
                    continue
                player = ranked.iloc[rank - 1]
                note_exclusion_keys.add(
                    f"{int(player['frame'])}:{clean_id(player['player_id'])}"
                )

    exclusion_keys.update(note_exclusion_keys)

    filtered = players[~players["_eligibility_key"].isin(exclusion_keys)].copy()
    removed = players[players["_eligibility_key"].isin(exclusion_keys)].copy()

    filtered = filtered.drop(columns=["_eligibility_key"])
    removed = removed.drop(columns=["_eligibility_key"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    removed_path = output_csv.with_name(output_csv.stem + "_removed.csv")
    removed.to_csv(removed_path, index=False)

    print("PLAYER ELIGIBILITY REVIEW FILTER COMPLETE")
    print("-----------------------------------------")
    print(f"Player detections: {len(players)}")
    print(f"Reviewed non-active exclusions: {len(exclusion_keys)}")
    print(f"Nearest-label exclusions: {len(set(exclusions['_eligibility_key']))}")
    print(f"Review-note rank exclusions: {len(note_exclusion_keys)}")
    print(f"Removed player detections: {len(removed)}")
    print(f"Remaining player detections: {len(filtered)}")
    print(f"Filtered players: {output_csv}")
    print(f"Removed detections: {removed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove frame/player detections marked as non-active in review."
    )
    parser.add_argument("player_csv")
    parser.add_argument("review_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--association-csv",
        default=None,
        help=(
            "Optional association CSV used to resolve ranked players mentioned "
            "in review_notes, like '#4 is referee'."
        ),
    )
    parser.add_argument("--nearest-limit", type=int, default=5)
    parser.add_argument(
        "--non-active-labels",
        default="referee,substitute_or_staff,off_field_player",
        help="Comma-separated eligibility labels to remove.",
    )
    args = parser.parse_args()

    apply_player_eligibility_reviews(
        player_csv=args.player_csv,
        review_csv=args.review_csv,
        output_csv=args.output,
        association_csv=args.association_csv,
        nearest_limit=args.nearest_limit,
        non_active_labels=[
            label.strip()
            for label in args.non_active_labels.split(",")
            if label.strip()
        ],
    )


if __name__ == "__main__":
    main()
