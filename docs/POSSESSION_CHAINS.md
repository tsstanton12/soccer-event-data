# Possession Chains

Frame-level nearest-player association is useful evidence, but it is not stable
enough to represent possession directly. During passes, the ball can pass near
an opponent and briefly make that opponent the nearest player. Treating that as
ownership creates false turnovers.

`src/52_build_possession_chains.py` adds a first possession-state layer on top
of `src/41_ball_player_association_v2.py`.

## V1 Logic

- Use `ball_state == controlled` plus a maximum player distance as control
  evidence.
- Require the same new player to be the control candidate for several
  consecutive frames before switching possession.
- While a new candidate is not confirmed, keep the previous controller and mark
  the row as `in_transit`.
- Preserve `loose_or_unclear` and `in_transit` states instead of assigning each
  frame to the nearest player.

Default parameters:

- `min_confirm_frames = 5`
- `max_control_distance_px = 80`

This is a conservative state machine, not a final possession model. It is
designed to suppress obvious pass-in-flight ownership errors and produce
segments that can be reviewed against video.

## Outputs

Frame output adds:

- `possession_state`
- `possession_player_id`
- `possession_segment_id`
- `pending_player_id`
- `pending_confirm_frames`
- `transition_reason`

Segment output contains one row per confirmed controller interval:

- `segment_id`
- `player_id`
- `start_time`
- `end_time`
- `duration_seconds`
- `start_frame`
- `end_frame`
- `end_reason`

## Next Improvements

- Add team identity and prevent same-team pass candidates from being treated as
  turnovers.
- Use ball direction and speed decay to distinguish pass reception from the ball
  merely passing near a player.
- Evaluate against reviewed SoccerTrack possession labels.
- Add a review overlay for confirmed possession chains.

## Initial Runs

With the default V1 settings:

| Clip | Association rows | Raw controlled nearest-player switches | Confirmed possession segments |
| --- | ---: | ---: | ---: |
| Army short clip | 4,118 | 2,321 | 17 |
| Le Moyne short clip | 4,418 | 2,490 | 20 |

These counts are not final accuracy metrics. They show that the state-machine
layer suppresses frame-to-frame nearest-player flicker and produces a smaller
set of possession intervals suitable for video review.

