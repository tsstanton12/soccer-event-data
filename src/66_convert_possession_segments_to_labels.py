#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_segments_to_labels(
    segments_csv,
    output_json,
    match_id,
    clip,
    half=1,
    team="unknown",
    confidence=0.25,
):
    segments_csv = Path(segments_csv)
    output_json = Path(output_json)

    segments = pd.read_csv(segments_csv).sort_values(["start_time", "end_time"])
    required = ["segment_id", "player_id", "start_time", "end_time"]
    missing = [column for column in required if column not in segments.columns]
    if missing:
        raise ValueError(f"Missing required segment columns: {missing}")

    output_segments = []
    for _, row in segments.iterrows():
        start = float(row["start_time"])
        end = float(row["end_time"])
        if end <= start:
            continue
        player = str(row["player_id"])
        output_segments.append({
            "id": str(row["segment_id"]),
            "start": start,
            "end": end,
            "state": "controlled",
            "team": team,
            "player": player,
            "confidence": confidence,
            "half": half,
            "source_action_id": "",
            "notes": (
                "Auto-derived possession chain segment. Team identity is "
                "provisional unless supplied by a team-classification layer."
            ),
            "source_segment_end_reason": str(row.get("end_reason", "")),
            "source_start_frame": int(row["start_frame"]) if "start_frame" in row and pd.notna(row["start_frame"]) else None,
            "source_end_frame": int(row["end_frame"]) if "end_frame" in row and pd.notna(row["end_frame"]) else None,
        })

    payload = {
        "schema_version": "1.0",
        "source": str(segments_csv),
        "source_type": "auto_possession_chain_segments_csv",
        "review_status": "auto_provisional",
        "match_id": match_id,
        "clip": clip,
        "half": half,
        "team_identity_status": "unknown" if team == "unknown" else "supplied_constant",
        "segments": output_segments,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("POSSESSION SEGMENTS CONVERTED")
    print("-----------------------------")
    print(f"Input segments: {segments_csv}")
    print(f"Input rows: {len(segments)}")
    print(f"Output segments: {len(output_segments)}")
    print(f"Team value: {team}")
    print(f"Output JSON: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert possession-chain segment CSVs into possession-label JSON."
    )
    parser.add_argument("segments_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument(
        "--team",
        default="unknown",
        help=(
            "Team label to assign to all segments. Use the default 'unknown' "
            "until a real team-classification layer exists."
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    convert_segments_to_labels(
        segments_csv=args.segments_csv,
        output_json=args.output,
        match_id=args.match_id,
        clip=args.clip,
        half=args.half,
        team=args.team,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
