#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


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


def remap_association_to_tracks(association_csv, tracked_players_csv, output_csv):
    association_csv = Path(association_csv)
    tracked_players_csv = Path(tracked_players_csv)
    output_csv = Path(output_csv)

    association = pd.read_csv(association_csv)
    tracked = pd.read_csv(tracked_players_csv)

    required_association = ["frame", "nearest_player_id"]
    required_tracks = ["frame", "player_id", "detector_player_id", "track_id"]
    missing_association = [c for c in required_association if c not in association.columns]
    missing_tracks = [c for c in required_tracks if c not in tracked.columns]
    if missing_association:
        raise ValueError(f"Association CSV missing columns: {missing_association}")
    if missing_tracks:
        raise ValueError(f"Tracked player CSV missing columns: {missing_tracks}")

    association = association.copy()
    tracked = tracked.copy()
    association["frame"] = association["frame"].astype(int)
    tracked["frame"] = tracked["frame"].astype(int)
    association["_nearest_player_id_clean"] = association["nearest_player_id"].map(clean_id)
    tracked["_detector_player_id_clean"] = tracked["detector_player_id"].map(clean_id)

    mapping = tracked[
        [
            "frame",
            "_detector_player_id_clean",
            "track_id",
            "track_age_frames",
            "track_status",
        ]
    ].drop_duplicates(["frame", "_detector_player_id_clean"], keep="first")

    remapped = association.merge(
        mapping,
        how="left",
        left_on=["frame", "_nearest_player_id_clean"],
        right_on=["frame", "_detector_player_id_clean"],
    )
    remapped["detector_nearest_player_id"] = remapped["nearest_player_id"]
    remapped["nearest_track_id"] = remapped["track_id"]
    has_track = remapped["nearest_track_id"].notna()
    remapped.loc[has_track, "nearest_player_id"] = remapped.loc[has_track, "nearest_track_id"].astype(int)
    remapped["nearest_player_id_source"] = has_track.map({
        True: "persistent_track_id",
        False: "unmapped_detector_id",
    })
    remapped = remapped.drop(
        columns=[
            "_nearest_player_id_clean",
            "_detector_player_id_clean",
            "track_id",
        ],
        errors="ignore",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    remapped.to_csv(output_csv, index=False)

    print("ASSOCIATION TRACK REMAP COMPLETE")
    print("--------------------------------")
    print(f"Association: {association_csv}")
    print(f"Tracked players: {tracked_players_csv}")
    print(f"Rows: {len(remapped)}")
    print(f"Mapped nearest-player rows: {int(has_track.sum())}")
    print(f"Unmapped nearest-player rows: {int((~has_track).sum())}")
    print(f"Output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace frame-local nearest_player_id with persistent track_id."
    )
    parser.add_argument("association_csv")
    parser.add_argument("tracked_players_csv")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    remap_association_to_tracks(
        association_csv=args.association_csv,
        tracked_players_csv=args.tracked_players_csv,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
