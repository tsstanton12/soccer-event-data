#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_LABELS = {"correct", "wrong_team"}
REJECT_LABELS = {"wrong_player", "should_be_unknown", "unclear"}
NON_LIVE_HINTS = ["goal_stoppage", "restart", "dead ball", "dead-ball"]
NO_CONTROL_HINTS = ["long pass", "ball is in the air", "far from any player"]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_id(value):
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def classify_review(row):
    label = clean_text(row.get("review_label")).lower()
    notes = clean_text(row.get("review_notes")).lower()
    corrected_team = clean_text(row.get("corrected_team"))
    assigned_team = clean_text(row.get("assigned_team"))

    if any(hint in notes for hint in NON_LIVE_HINTS):
        return "reject_non_live", "unknown"
    if any(hint in notes for hint in NO_CONTROL_HINTS):
        return "reject_not_controlled", "unknown"
    if label == "correct":
        return "accepted", assigned_team or "unknown"
    if label == "wrong_team" and corrected_team:
        return "accepted_corrected_team", corrected_team
    if label in REJECT_LABELS:
        return f"reject_{label}", "unknown"
    if label in VALID_LABELS:
        return "reject_missing_correction", "unknown"
    return "reject_unreviewed", "unknown"


def load_source_segments(source_labels_json):
    payload = json.loads(Path(source_labels_json).read_text())
    segments = payload.get("segments", [])
    by_key = {}
    for segment in segments:
        by_key[(payload.get("clip", ""), segment.get("id", ""))] = segment
        by_key[(segment.get("id", ""),)] = segment
    return payload, segments, by_key


def main():
    parser = argparse.ArgumentParser(
        description="Apply team-classification review labels to possession segments."
    )
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--source-labels-json", action="append", required=True)
    parser.add_argument("--output-segments", required=True)
    parser.add_argument("--output-labels-json", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip", default="multi_clip_review")
    parser.add_argument("--half", type=int, default=1)
    args = parser.parse_args()

    reviews = pd.read_csv(args.review_manifest)
    source_segments = {}
    source_payloads = []
    for labels_json in args.source_labels_json:
        payload, _segments, by_key = load_source_segments(labels_json)
        source_payloads.append(payload)
        for key, value in by_key.items():
            source_segments[key] = value

    rows = []
    output_segments = []
    for _, review in reviews.iterrows():
        status, final_team = classify_review(review)
        clip_name = clean_text(review.get("video"))
        segment_id = clean_text(review.get("segment_id"))
        source_segment = (
            source_segments.get((clip_name, segment_id))
            or source_segments.get((segment_id,))
            or {}
        )
        accepted = status.startswith("accepted")
        row = review.to_dict()
        row.update({
            "review_status_applied": status,
            "final_team": final_team,
            "accepted_for_event_derivation": accepted,
        })
        rows.append(row)

        if not accepted:
            continue
        output_segments.append({
            "id": segment_id,
            "start": float(source_segment.get("start", review["start_time"])),
            "end": float(source_segment.get("end", review["end_time"])),
            "state": "controlled",
            "team": final_team,
            "player": clean_id(source_segment.get("player", review.get("player_id"))),
            "confidence": float(source_segment.get("confidence", review.get("team_vote_share", 0.5))),
            "half": int(source_segment.get("half", args.half)),
            "source_action_id": source_segment.get("source_action_id", ""),
            "notes": (
                "Team classification reviewed by human; player identity remains "
                "frame-local unless separately corrected."
            ),
            "source_segment_end_reason": source_segment.get("source_segment_end_reason", ""),
            "team_review_label": clean_text(review.get("review_label")),
            "team_review_notes": clean_text(review.get("review_notes")),
        })

    applied = pd.DataFrame(rows)
    output_segments_path = Path(args.output_segments)
    output_segments_path.parent.mkdir(parents=True, exist_ok=True)
    applied.to_csv(output_segments_path, index=False)

    payload = {
        "schema_version": "1.0",
        "source_type": "reviewed_team_classification_segments",
        "review_status": "team_review_applied",
        "match_id": args.match_id,
        "clip": args.clip,
        "half": args.half,
        "team_identity_status": "human_reviewed",
        "player_identity_status": "frame_local_not_persistent",
        "segments": sorted(output_segments, key=lambda row: (row["start"], row["end"])),
    }
    output_labels_path = Path(args.output_labels_json)
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("TEAM CLASSIFICATION REVIEWS APPLIED")
    print("-----------------------------------")
    print(f"Review rows: {len(applied)}")
    print("Applied status:")
    print(applied["review_status_applied"].value_counts().to_string())
    print(f"Accepted segments: {len(output_segments)}")
    print(f"Applied CSV: {output_segments_path}")
    print(f"Labels JSON: {output_labels_path}")


if __name__ == "__main__":
    main()
