# Event Schema

Version `0.1` is intentionally small. It captures event candidates that can be
derived from reviewed possession labels before team identity, coordinates, and
tracking-quality signals are fully integrated.

## Possession Label

```json
{
  "id": "segment_0001",
  "start": 0.68,
  "end": 2.12,
  "state": "controlled",
  "team": "right",
  "player": "467351",
  "confidence": 0.5,
  "notes": "Reviewed as correct; derived from PASS",
  "half": 1,
  "source_action_id": "action_0001"
}
```

Rules:

- `start` and `end` are seconds within the selected half video.
- `state` is one of `controlled`, `contested`, `loose`, `airborne`,
  `out_of_play`, or `unknown`.
- `controlled` segments require `team` and `player`.
- Ambiguous control should use a non-controlled state rather than forcing a
  player.

## Completed Pass Candidate

```json
{
  "id": "event_0001",
  "schema_version": "0.1",
  "match_id": "117093",
  "clip": "117093_panorama_1st_half_review_0000_0060.mp4",
  "half": 1,
  "type": "completed_pass_candidate",
  "team": "left",
  "from_player": "200211",
  "to_player": "266083",
  "start": 16.68,
  "end": 17.52,
  "duration": 0.84,
  "from_possession_id": "segment_0007",
  "to_possession_id": "segment_0008",
  "from_source_action_id": "action_0008",
  "to_source_action_id": "action_0009",
  "from_source_note": "Reviewed as correct; derived from PASS",
  "confidence": 0.5,
  "review_status": "candidate",
  "notes": "Derived from consecutive reviewed controlled possessions."
}
```

Derivation rule:

- two consecutive controlled possession segments;
- same `team`;
- different `player`;
- same `half`;
- transition gap from previous end to next start is between `0` and the
  configured maximum, currently `2.0` seconds.
- if reviewed game-state intervals are supplied, neither possession segment nor
  the transition between them may overlap a non-live interval.

`src/51_derive_events_from_possession.py` accepts optional reviewed game-state
intervals:

```bash
.venv/bin/python src/51_derive_events_from_possession.py \
  data/soccertrack/117093/reviewed/117093_1st_half_review_0000_0060_possession_labels.json \
  --output data/soccertrack/117093/derived/117093_1st_half_review_0000_0060_event_candidates.json \
  --game-state-intervals outputs/game_state_review/<matching_clip>/reviewed_game_state_intervals.csv
```

The output JSON includes:

- `event_count`
- `skipped_event_count`
- `events`
- `skipped_events`

Skipped events include a `reason`, `game_state_label`, `restart_type`, and
`source_review_id` so a reviewer can see why they were removed.

This still produces candidates. Later versions should distinguish true passes
from deflections, touches, carries, set-piece restarts, and tracking artifacts.
