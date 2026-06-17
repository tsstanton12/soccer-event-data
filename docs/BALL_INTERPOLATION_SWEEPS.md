# Ball Interpolation Sweeps

Interpolation is a major control point for this project. It can bridge missing
ball detections caused by camera quality, but it can also invent ball motion and
create false possession control.

Use `src/83_sweep_ball_interpolation.py` to test interpolation settings before
changing the main possession pipeline.

## Example Commands

LeMoyne smoke sweep:

```sh
.venv/bin/python src/83_sweep_ball_interpolation.py \
  outputs/lemoyne/lemoyne_short_clip_conservative_ball_path.csv \
  --output-dir outputs/ball_interpolation_sweep/lemoyne_smoke \
  --fps 30 \
  --max-gap-seconds 0.5,1.0 \
  --max-speeds 1800,2500 \
  --players-csv outputs/player_tracking/lemoyne_players_on_field_tracked.csv \
  --track-stability-csv outputs/player_tracking/track_stability/lemoyne_min15/track_stability_summary.csv
```

Army smoke sweep:

```sh
.venv/bin/python src/83_sweep_ball_interpolation.py \
  outputs/army/army_short_clip_conservative_ball_path.csv \
  --output-dir outputs/ball_interpolation_sweep/army_smoke \
  --fps 25 \
  --max-gap-seconds 0.5,1.0 \
  --max-speeds 1800,2500 \
  --players-csv outputs/player_tracking/army_players_on_field_tracked.csv \
  --track-stability-csv outputs/player_tracking/track_stability/army_min15/track_stability_summary.csv
```

## What To Watch

Good interpolation should:

- increase ball-path coverage;
- reduce short remaining gaps;
- avoid a large interpolated share;
- avoid suspicious speed jumps;
- avoid creating extra possession segments from invented close-to-player ball
  positions.

The summary CSV and report are written inside the chosen output directory.

## First Smoke-Test Result

Smoke tests on the current conservative Army and LeMoyne ball paths showed:

- changing `--max-speed` from `1800` to `2500` did not change either clip,
  which means those speed gates were not binding on the current gaps;
- increasing `--max-gap-seconds` from `0.5` to `1.0` substantially increased
  interpolated rows;
- the 1.0 second setting also increased confirmed possession segments, which
  means it may be inventing enough ball continuity to affect ownership.

This argues for treating max gap length as the first interpolation parameter to
tune. The speed gate is still useful, but these clips need more aggressive
tests before it becomes the limiting factor.
