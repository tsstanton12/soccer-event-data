# Ball Model Rerun Workflow

Use this after a new Roboflow-trained ball model is exported.

## One-Command Rerun

`src/79_rerun_ball_model_pipeline.py` runs the full clip workflow once a new
model is ready.

LeMoyne example:

```sh
.venv/bin/python src/79_rerun_ball_model_pipeline.py \
  --venue-label lemoyne \
  --video videos/lemoyne_short_clip.mp4 \
  --players outputs/player_tracking/lemoyne_players_on_field_tracked.csv \
  --track-stability outputs/player_tracking/track_stability/lemoyne_min15/track_stability_summary.csv \
  --fps 30 \
  --model path/to/roboflow_best.pt
```

Army example:

```sh
.venv/bin/python src/79_rerun_ball_model_pipeline.py \
  --venue-label army \
  --video videos/army_short_clip.mp4 \
  --players outputs/player_tracking/army_players_on_field_tracked.csv \
  --track-stability outputs/player_tracking/track_stability/army_min15/track_stability_summary.csv \
  --fps 25 \
  --model path/to/roboflow_best.pt
```

Add `--dry-run` first if you only want to preview the commands and output
paths.

The script writes into `outputs/<venue-label>_new_ball_model/` by default.

## 1. Evaluate The New Model On The Hard Batch

First run the hard-frame workflow in `docs/BALL_MODEL_EVALUATION.md`. Do not
rerun full possession chains until the hard-frame evaluation looks better than
the current model. The hard batch is designed around the errors that matter
most for possession: missed balls, wrong balls, suspicious jumps, and
low-confidence detections.

## 2. Run The New Model On A Full Clip

Example for LeMoyne:

```sh
.venv/bin/python src/08_detect_ball_custom_model.py \
  --venue lemoyne_new_ball_model \
  --video videos/lemoyne_short_clip.mp4 \
  --model path/to/roboflow_best.pt \
  --conf 0.20
```

Example for Army:

```sh
.venv/bin/python src/08_detect_ball_custom_model.py \
  --venue army_new_ball_model \
  --video videos/army_short_clip.mp4 \
  --model path/to/roboflow_best.pt \
  --conf 0.20
```

This writes detections to:

- `outputs/lemoyne_new_ball_model/lemoyne_short_clip_ball_detections.csv`
- `outputs/army_new_ball_model/army_short_clip_ball_detections.csv`

## 3. Interpolate Conservative Gaps

After selecting/filtering the preferred ball path, use cautious interpolation:

```sh
.venv/bin/python src/38_interpolate_ball_path_extended.py \
  outputs/lemoyne_new_ball_model/lemoyne_short_clip_ball_detections.csv \
  --output outputs/lemoyne_new_ball_model/lemoyne_short_clip_extended_interpolated_ball_path.csv \
  --fps 30 \
  --max-gap-seconds 1.0 \
  --max-speed 2500
```

Use `--fps 25` for the Army short clip.

## 4. Rebuild Ball-Player Association

Use the tracked player file, not the frame-local detector IDs:

```sh
.venv/bin/python src/41_ball_player_association_v2.py \
  outputs/lemoyne_new_ball_model/lemoyne_short_clip_extended_interpolated_ball_path.csv \
  outputs/player_tracking/lemoyne_players_on_field_tracked.csv \
  --output outputs/lemoyne_new_ball_model/lemoyne_ball_player_association_tracked.csv \
  --fps 30 \
  --control-field-zones strict,tolerant \
  --recompute-time-seconds
```

Use the matching Army paths and `--fps 25` for Army.

## 5. Rebuild Possession Chains

Use the min-15 confirmation setting plus the current stability gate:

```sh
.venv/bin/python src/52_build_possession_chains.py \
  outputs/lemoyne_new_ball_model/lemoyne_ball_player_association_tracked.csv \
  --output-frames outputs/lemoyne_new_ball_model/lemoyne_min15_stability_gated_frames.csv \
  --output-segments outputs/lemoyne_new_ball_model/lemoyne_min15_stability_gated_segments.csv \
  --fps 30 \
  --min-confirm-frames 15 \
  --max-control-distance-px 80 \
  --max-carry-forward-seconds 2.0 \
  --track-stability-csv outputs/player_tracking/track_stability/lemoyne_min15/track_stability_summary.csv \
  --exclude-track-stability-flags short_track,jumpy_track,gappy_track
```

Then render a first-minute overlay:

```sh
.venv/bin/python src/53_visualize_possession_chains.py \
  videos/lemoyne_short_clip.mp4 \
  outputs/lemoyne_new_ball_model/lemoyne_min15_stability_gated_frames.csv \
  --output outputs/lemoyne_new_ball_model/lemoyne_min15_stability_gated_overlay_first60s.mp4 \
  --max-frames 1800
```

## Current Caution

`src/11_filter_ball_detections.py` is an older hard-coded script and should not
be treated as the canonical rerun path without refactoring. For now, prefer the
hard-batch evaluation plus cautious full-clip reruns above.
