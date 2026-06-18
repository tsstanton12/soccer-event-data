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

The reset roadmap is now tracked separately in `docs/PROJECT_ROADMAP.md`.
The hard-frame ball model evaluation workflow is documented in
`docs/BALL_MODEL_EVALUATION.md`.
The full-clip rerun workflow for a new ball model is documented in
`docs/BALL_MODEL_RERUN.md`.
The ball interpolation sweep workflow is documented in
`docs/BALL_INTERPOLATION_SWEEPS.md`.
The current human-review queue is tracked in `docs/MANUAL_NEXT_STEPS.md`.
The latest off-repository ChatGPT/Codex handoff is captured in
`docs/HANDOFF_2026_06_18.md`.

## Latest Handoff Status

As of the 2026-06-18 handoff:

- The Roboflow ball annotation batch is complete and a new YOLOv8 model is
  training in Colab. The latest reported artifact name is `best_june_17.pt`;
  exported YOLO labels are optional for the first model-only overlay rerun.
- The LeMoyne 65-image non-active participant review is complete. A conservative
  combined filter excludes tracks `145` and `114` as referees and track `126` as
  substitute/staff.
- Mixed or sparse tracks such as `103`, `130`, `117`, `177`, and `23` should not
  be auto-removed without further review.
- The next visual review is the LeMoyne combined-filter overlay compared with
  the previous track-filtered overlay.
- Local scripts `src/84_extract_ranked_eligibility_notes.py` and
  `src/85_merge_eligibility_track_summaries.py` were reported in the handoff,
  but may need to be ported into this checkout if that workflow is needed here.

## Current Pipeline

The numbered scripts in `src/` represent the working pipeline:

- Scripts 02-16: player/ball detection, filtering, trajectory generation, and
  review.
- Scripts 17-38: field masks, conservative ball paths, progression detection,
  calibration, interpolation, and diagnostics.
- Scripts 39-43: ball-player association and on-field player filtering.
- Scripts 44-50: manual/automatic playable-field masks, segmentation training,
  evaluation, and failure-batch review.
- Scripts 51-71: reviewed-possession event derivation, possession-chain
  smoothing, event-impact review, game-state review, reviewed non-live window
  application, auto-segment conversion into event-derivation inputs,
  provisional jersey-color team classification, team-review application, and
  simple persistent player tracking/review overlays.

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

1. Review/evaluate simple persistent player tracks, then rerun ball-player
   association and possession chains with persistent `track_id` values instead
   of frame-local detector IDs.
2. Inspect SoccerNet Ball Action Spotting access and baselines using
   `docs/OPEN_DATA_AUGMENTATION.md`.
3. Train/evaluate automatic possession assignment before expanding event
   inference beyond completed-pass candidates.

## Simple Player Tracking

`src/70_track_players_simple.py` assigns persistent `track_id` values to
frame-level player detections using bounding-box overlap, foot-point distance,
and box-size consistency. It is intentionally lightweight: it does not use
appearance embeddings or a learned re-identification model, but it gives the
pipeline a stable identity column that is much better than the detector's
per-frame `player_id`.

Initial tracked outputs:

- `outputs/player_tracking/army_players_on_field_tracked.csv`
- `outputs/player_tracking/lemoyne_players_on_field_tracked.csv`
- `outputs/player_tracking/army_track_overlay_first60s.mp4`
- `outputs/player_tracking/lemoyne_track_overlay_first60s.mp4`

Both were generated with `--replace-player-id`, so the `player_id` column is
now the persistent track ID and the original per-frame detector ID is preserved
as `detector_player_id`. These tracks should be visually reviewed before using
them as final player identities, but they are the right next input for rerunning
ball-player association and possession chains.

`src/71_visualize_player_tracks.py` renders a video overlay with `T<track_id>`
labels so track continuity can be checked visually.

`src/72_remap_association_to_tracks.py` preserves tuned association/ball-state
rows while replacing frame-local `nearest_player_id` values with persistent
`track_id` values. This is preferable to rebuilding association from scratch
when reviewing the effect of persistent identity, because it keeps the
receiver-confirmed ball-state smoothing and active-participant context already
used in prior reviews.

Tracked-ID possession rerun outputs:

- `outputs/player_tracking/association_rerun/army_tuned_association_tracked_ids.csv`
- `outputs/player_tracking/association_rerun/army_tuned_tracked_possession_chain_frames.csv`
- `outputs/player_tracking/association_rerun/army_tuned_tracked_possession_chain_segments.csv`
- `outputs/player_tracking/association_rerun/army_tuned_tracked_possession_overlay_first60s.mp4`
- `outputs/player_tracking/association_rerun/lemoyne_tuned_association_tracked_ids.csv`
- `outputs/player_tracking/association_rerun/lemoyne_tuned_tracked_possession_chain_frames.csv`
- `outputs/player_tracking/association_rerun/lemoyne_tuned_tracked_possession_chain_segments.csv`
- `outputs/player_tracking/association_rerun/lemoyne_tuned_tracked_possession_overlay_first60s.mp4`

The first tracked-ID chain rerun produced many more segments than the previous
frame-local-ID reviewed candidates. That is expected to some degree: detector
IDs reset every frame and can accidentally hide controller switches. The
tracked overlays should be reviewed before using these segments for event
derivation.

Initial visual review found that many tracked-ID switches, especially in the
Army clip, are noisy nearest-player changes while the ball is traveling. The
nearest player may be correctly identified, but possession should not change
owners until a receiver has clearly controlled the ball. A first threshold sweep
using persistent IDs showed:

| Clip | `min_confirm_frames=5` | `8` | `12` | `15` | `20` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Army segments | 92 | 66 | 44 | 27 | 19 |
| LeMoyne segments | 72 | 54 | 39 | 30 | 28 |

Use the `min_confirm_frames=15` overlays as the next review candidate:

- `outputs/player_tracking/association_rerun/army_tuned_tracked_min15_possession_overlay_first60s.mp4`
- `outputs/player_tracking/association_rerun/lemoyne_tuned_tracked_min15_possession_overlay_first60s.mp4`

Follow-up visual review found:

- Owner switches during pass travel are reduced with `min_confirm_frames=15`.
- Army remains somewhat jumpy, but many remaining incorrect switches appear to
  be caused by bad ball detections rather than bad player tracking.
- LeMoyne is close to the real possession flow. Remaining errors are mostly
  non-active people such as substitutes or assistant referees, plus occasional
  one-touch possessions that stay `in_transit` through the touch.

Focused review clips were generated from the min-15 tracked chains:

- `outputs/player_tracking/association_rerun/min15_event_impact_review/army_first60/event_impact_review_order.html`
- `outputs/player_tracking/association_rerun/min15_event_impact_review/lemoyne_first60/event_impact_review_order.html`

The next likely fixes are targeted ball-detection review/improvement for Army
and stronger non-active participant filtering for LeMoyne before deriving
events from the tracked chains.

`src/77_evaluate_player_track_stability.py` now evaluates persistent player
tracks and writes a per-track summary, a possession-relevant trouble-track
list, and a markdown report.

Current track-stability outputs:

- `outputs/player_tracking/track_stability/lemoyne_min15/`
- `outputs/player_tracking/track_stability/army_min15/`

The first diagnostic pass supports the visual review: LeMoyne is mostly stable
where possession is concerned, while Army still has a meaningful unstable-track
component. LeMoyne had 241 possession rows tied to non-stable tracks out of
2,927 possession rows. Army had 347 possession rows tied to non-stable tracks
out of 1,592 possession rows, plus 659 controlled-nearest rows tied to
non-stable tracks. That means Army's remaining possession noise is not only a
ball-detection problem; it also needs track-stability gating before we trust
ownership.
