#!/usr/bin/env python3

import argparse
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


def derive_completed_passes(payload, segments, max_gap_seconds):
    controlled = [
        segment
        for segment in segments
        if segment.get("state") == "controlled"
        and segment.get("team")
        and segment.get("player")
    ]

    events = []
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

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Derive event candidates from reviewed possession labels."
    )
    parser.add_argument("labels_json")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    args = parser.parse_args()

    payload, segments = load_segments(args.labels_json)
    validate_segments(segments)
    events = derive_completed_passes(payload, segments, args.max_gap_seconds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "0.1",
        "source_labels": str(args.labels_json),
        "match_id": payload.get("match_id", ""),
        "clip": payload.get("clip", ""),
        "half": payload.get("half", ""),
        "event_count": len(events),
        "events": events,
    }, indent=2) + "\n")

    print("POSSESSION EVENT DERIVATION COMPLETE")
    print("------------------------------------")
    print(f"Input labels: {args.labels_json}")
    print(f"Segments: {len(segments)}")
    print(f"Completed pass candidates: {len(events)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
