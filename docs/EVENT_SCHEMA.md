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

## Auto Possession Segment Path

`src/66_convert_possession_segments_to_labels.py` converts possession-chain CSV
segments into the same JSON label shape consumed by
`src/51_derive_events_from_possession.py`.

Example:

```bash
.venv/bin/python src/66_convert_possession_segments_to_labels.py \
  outputs/event_impact_review/army_first60_tuned/army_event_impact_candidate_start_stability_segments.csv \
  --output outputs/game_state_review/army_from_event_impact/army_auto_possession_labels.json \
  --match-id army_short_clip \
  --clip army_short_clip.mp4 \
  --team unknown
```

When team identity is `unknown`, event derivation is conservative by default:
adjacent controlled possessions are skipped with reason `unknown_team_identity`.
Use `--allow-unknown-team` only to emit provisional
`possession_transition_candidate` rows for review. These are not completed pass
candidates until team classification confirms the two possessions belong to the
same team.

`src/67_classify_team_from_jersey_color.py` adds the first automatic team
classification layer for broadcast clips. It samples player crops from
controlled possession segments, clusters jersey colors into two provisional
teams, votes those labels onto possession segments, and writes a possession
label JSON that can be passed directly to event derivation.

Example:

```bash
.venv/bin/python src/67_classify_team_from_jersey_color.py \
  --video videos/army_short_clip.mp4 \
  --players outputs/army/army_short_clip_player_detections.csv \
  --possession-frames outputs/game_state_review/army_from_event_impact/army_event_impact_candidate_with_game_state_frames.csv \
  --possession-segments outputs/event_impact_review/army_first60_tuned/army_event_impact_candidate_start_stability_segments.csv \
  --output-players outputs/team_classification/army_from_event_impact/army_player_team_classification.csv \
  --output-frames outputs/team_classification/army_from_event_impact/army_possession_frames_with_team.csv \
  --output-segments outputs/team_classification/army_from_event_impact/army_possession_segments_with_team.csv \
  --output-labels-json outputs/team_classification/army_from_event_impact/army_auto_possession_labels_with_team.json \
  --match-id army_short_clip \
  --clip army_short_clip.mp4
```

The resulting team labels are provisional: `team_1` and `team_2` mean jersey
color clusters, not known school/team names. Current player IDs are still
frame-local detector IDs, so pass candidates from auto possession remain
review-required until persistent player identity is added.

This still produces candidates. Later versions should distinguish true passes
from deflections, touches, carries, set-piece restarts, and tracking artifacts.
