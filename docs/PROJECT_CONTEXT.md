# Project Context

This document consolidates the useful decisions and status from the earlier
Auto Field Detection, Add Possession Assignment, and Strict/Tolerant Field
Polygon conversations. This repository is now the canonical workspace.

## Goal

Turn broadcast soccer video into structured event data:

1. Detect and track players and the ball.
2. Determine which detections are on the playable field.
3. Associate the ball with a likely controlling player.
4. Infer and review possession segments.
5. Derive events such as completed passes.
6. Map events into field coordinates for downstream analysis such as xT.

## Current Pipeline

The numbered scripts in `src/` represent the working pipeline:

- Scripts 02-16: player/ball detection, filtering, trajectory generation, and
  review.
- Scripts 17-38: field masks, conservative ball paths, progression detection,
  calibration, interpolation, and diagnostics.
- Scripts 39-43: ball-player association and on-field player filtering.
- Scripts 44-50: manual/automatic playable-field masks, segmentation training,
  evaluation, and failure-batch review.

## Playable-Field Decisions

The second playable-field segmentation model was reviewed on 48 frames:

- 43 good: 89.6%
- 5 too tight: 10.4%
- 0 too wide
- 0 missed

All failures were conservative edge cutoffs, usually involving a goalkeeper.
The annotations should remain on the true playable-field boundary rather than
being deliberately widened.

The adopted downstream design uses two boundaries:

- **Strict polygon:** the model prediction, used to avoid associating the ball
  with coaches, substitutes, and other off-field people.
- **Tolerant polygon:** the strict polygon expanded by approximately 15 pixels,
  used to retain legitimate edge players.

This behavior is implemented:

- `src/45_auto_field_mask.py` emits strict and tolerant polygons. Segmentation
  inset defaults to `0`.
- `src/43_filter_on_field_players.py` classifies player foot points as
  `strict`, `tolerant`, or `off_field` using the nearest previous frame mask.
- `src/41_ball_player_association_v2.py` prefers strict players when one is
  within association range, otherwise permits tolerant players.

The strict/tolerant behavior was evaluated on the 48-frame V2 review set. The
current `0px` segmentation inset and `15px` tolerant margin retained 5,120
tolerant-only Army detections while preserving the desired exclusion of
sideline figures. See `docs/STRICT_TOLERANT_EVALUATION.md`.

## Possession Lab

`tools/possession-lab/` is a standalone browser tool for:

- importing SoccerTrack v2 BAS action annotations;
- deriving conservative, review-required possession segments;
- manually labeling possession from video;
- evaluating predicted possession against reviewed labels;
- generating initial completed-pass candidates.

For each automatically derived possession segment, review:

- controlling player;
- team;
- start and end time;
- state: `controlled`, `contested`, `loose`, `airborne`, `out_of_play`, or
  `unknown`;
- whether a listed teammate-to-teammate transition is genuinely a pass.

Do not force ambiguous moments into player possession.

## SoccerTrack v2 Inputs

Start with one match. For a match ID such as `117092`, download:

```text
bas/117092/117092_12_class_events.json
videos/117092/117092_panorama_1st_half.mp4
videos/117092/117092_panorama_2nd_half.mp4
```

Later, add:

```text
gsr/117092/117092_1st.json
gsr/117092/117092_2nd.json
```

The BAS file provides action evidence. GSR provides per-frame player
positions, teams, and persistent IDs for future automatic player-possession
training. BAS timestamps restart at zero for each half, so review one half at a
time.

SoccerTrack v2 is CC BY 4.0; preserve attribution in derived data.

## Next Work

1. Use Possession Lab to review SoccerTrack-derived segments for one match.
2. Define a stable common event/possession schema shared by the video pipeline,
   Possession Lab, and `soccer-xT-project`.
3. Add team classification and persistent player identity to the broadcast
   video pipeline.
4. Train/evaluate automatic possession assignment before expanding event
   inference beyond completed-pass candidates.
