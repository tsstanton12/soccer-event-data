#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def load_segments(path):
    payload = json.loads(Path(path).read_text())
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Possession labels must contain a segments array.")
    return payload, sorted(segments, key=lambda row: (row.get("half", 1), row["start"]))


def validate_segments(segments):
    errors = []
    for index, segment in enumerate(segments, start=1):
        for key in ["start", "end", "state"]:
            if key not in segment:
                errors.append(f"segment {index} missing {key}")
        if segment.get("end", 0) <= segment.get("start", 0):
            errors.append(f"segment {index} end must be after start")
        if segment.get("state") == "controlled":
            if not segment.get("team"):
                errors.append(f"segment {index} controlled segment missing team")
            if not segment.get("player"):
                errors.append(f"segment {index} controlled segment missing player")
    if errors:
        raise ValueError("\n".join(errors))


def overlaps(start_a, end_a, start_b, end_b):
    return start_a <= end_b and start_b <= end_a


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_non_live_intervals(path):
    if not path:
        return []

    rows = list(csv.DictReader(Path(path).open()))
    intervals = []
    for row in rows:
        if "start_time" in row and "end_time" in row:
            is_live = parse_bool(row.get("is_live", row.get("game_state_is_live", "true")))
            label = row.get("game_state_label", "unknown")
            if is_live:
                continue
            intervals.append({
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "label": label or "non_live",
                "restart_type": row.get("restart_type", "none") or "none",
                "source_review_id": row.get("source_review_id", ""),
            })
            continue

        if "time_seconds" in row and "game_state_is_live" in row:
            if parse_bool(row["game_state_is_live"]):
                continue
            time_seconds = float(row["time_seconds"])
            intervals.append({
                "start": time_seconds,
                "end": time_seconds,
                "label": row.get("game_state_label", "non_live") or "non_live",
                "restart_type": row.get("game_state_restart_type", "none") or "none",
                "source_review_id": row.get("game_state_source_review_id", ""),
            })

    return intervals


def find_non_live_overlap(start, end, non_live_intervals):
    for interval in non_live_intervals:
        if overlaps(start, end, interval["start"], interval["end"]):
            return interval
    return None


def derive_completed_passes(payload, segments, max_gap_seconds, non_live_intervals=None):
    if non_live_intervals is None:
        non_live_intervals = []

    controlled = [
        segment
        for segment in segments
        if segment.get("state") == "controlled"
        and segment.get("team")
        and segment.get("player")
    ]

    events = []
    skipped = []
    for index in range(len(controlled) - 1):
        current = controlled[index]
        next_segment = controlled[index + 1]
        gap = next_segment["start"] - current["end"]

        if current.get("half", payload.get("half", 1)) != next_segment.get(
            "half", payload.get("half", 1)
        ):
            continue
        if current["team"] != next_segment["team"]:
            continue
        if current["player"] == next_segment["player"]:
            continue
        if gap < 0 or gap > max_gap_seconds:
            continue

        non_live_overlap = find_non_live_overlap(
            current["start"],
            next_segment["end"],
            non_live_intervals,
        )
        if non_live_overlap:
            skipped.append({
                "from_possession_id": current.get("id", ""),
                "to_possession_id": next_segment.get("id", ""),
                "start": current["end"],
                "end": next_segment["start"],
                "reason": "non_live_game_state_overlap",
                "game_state_label": non_live_overlap["label"],
                "restart_type": non_live_overlap["restart_type"],
                "source_review_id": non_live_overlap["source_review_id"],
            })
            continue

        event_id = f"event_{len(events) + 1:04d}"
        source_note = current.get("notes", "")
        events.append({
            "id": event_id,
            "schema_version": "0.1",
            "match_id": payload.get("match_id", ""),
            "clip": payload.get("clip", ""),
            "half": current.get("half", payload.get("half", 1)),
            "type": "completed_pass_candidate",
            "team": current["team"],
            "from_player": current["player"],
            "to_player": next_segment["player"],
            "start": current["end"],
            "end": next_segment["start"],
            "duration": gap,
            "from_possession_id": current.get("id", ""),
            "to_possession_id": next_segment.get("id", ""),
            "from_source_action_id": current.get("source_action_id", ""),
            "to_source_action_id": next_segment.get("source_action_id", ""),
            "from_source_note": source_note,
            "confidence": min(
                float(current.get("confidence", 1)),
                float(next_segment.get("confidence", 1)),
            ),
            "review_status": "candidate",
            "notes": "Derived from consecutive reviewed controlled possessions.",
        })

    return events, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Derive event candidates from reviewed possession labels."
    )
    parser.add_argument("labels_json")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument(
        "--game-state-intervals",
        default="",
        help=(
            "Optional CSV from src/64_apply_game_state_reviews.py. Candidate "
            "events overlapping non-live reviewed windows are skipped."
        ),
    )
    args = parser.parse_args()

    payload, segments = load_segments(args.labels_json)
    validate_segments(segments)
    non_live_intervals = load_non_live_intervals(args.game_state_intervals)
    events, skipped_events = derive_completed_passes(
        payload,
        segments,
        args.max_gap_seconds,
        non_live_intervals=non_live_intervals,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "0.1",
        "source_labels": str(args.labels_json),
        "match_id": payload.get("match_id", ""),
        "clip": payload.get("clip", ""),
        "half": payload.get("half", ""),
        "event_count": len(events),
        "skipped_event_count": len(skipped_events),
        "game_state_intervals": str(args.game_state_intervals) if args.game_state_intervals else "",
        "events": events,
        "skipped_events": skipped_events,
    }, indent=2) + "\n")

    print("POSSESSION EVENT DERIVATION COMPLETE")
    print("------------------------------------")
    print(f"Input labels: {args.labels_json}")
    print(f"Segments: {len(segments)}")
    print(f"Non-live intervals: {len(non_live_intervals)}")
    print(f"Completed pass candidates: {len(events)}")
    print(f"Skipped by game state: {len(skipped_events)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
