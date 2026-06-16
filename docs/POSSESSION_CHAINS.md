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
- If the ball remains loose or in transit beyond the carry-forward timeout, end
  the active possession instead of assigning ownership indefinitely to the last
  confirmed player.
- Preserve `loose_or_unclear` and `in_transit` states instead of assigning each
  frame to the nearest player.

Default parameters:

- `min_confirm_frames = 5`
- `max_control_distance_px = 80`
- `max_carry_forward_seconds = 2.0`

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
- `carry_forward_frames`
- `carry_forward_seconds`
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
- Review more overlay clips and tune the carry-forward timeout by match context.

## Review Overlay

`src/53_visualize_possession_chains.py` renders an MP4 with the raw
nearest-player evidence and the smoothed possession-chain state overlaid on the
video. The overlay is intended for quick visual QA of threshold choices.

The ball marker color follows the raw `ball_state`: green means the ball is
classified as `controlled`, orange means `in_transit`, red means
`loose_or_unclear`, and gray means `unknown`. The panel separately shows the
smoothed possession-chain state and confirmed/pending owner.

Raw ball-state classification treats high ball speed as transit before applying
the normal player-distance cutoff. This keeps longer passes orange even when the
ball is temporarily farther than the association radius from every player.

`src/54_build_ball_state_transition_review.py` exports side-by-side review
images for raw ball-state transitions. Each image pairs the prior model row with
the row where `ball_state` changes, so a reviewer can mark whether the switch
between `controlled`, `in_transit`, and `loose_or_unclear` is correct. The
manifest includes blank `review_correct` and `review_notes` columns for
feedback.

`src/55_evaluate_ball_state_reviews.py` converts a completed transition-review
manifest into frame-level labels and scores a candidate association CSV against
those labels. On the first completed Le Moyne review, the current raw classifier
matched 44 of 168 reviewed frame labels.

`src/56_smooth_ball_state_intervals.py` is an experimental second-pass smoother
that treats pass travel as an interval instead of independent frame decisions.
The first candidate raised the Le Moyne review score to 124 of 168 labels, but
it over-labels some reviewed `controlled` and `loose_or_unclear` moments as
`in_transit`, so it should remain a tuning candidate rather than the production
default for now.

A second review of the smoothed Le Moyne candidate marked 27 of 54 transitions
correct. Converted to frame labels, the candidate matched 78 of 108 reviewed
states. The remaining errors clustered into:

- far-side ball detection mistakes, including player cleats being detected as
  the ball;
- sideline/referee/non-player detections becoming the nearest player;
- passes traveling close to defenders and being ended as `controlled` before a
  real touch;
- a few early-clip frames that should remain `controlled`.

This suggests the next model improvement should add receiver confirmation for
ending `in_transit`, and separately improve player filtering near sidelines.

A receiver-confirmation candidate requiring at least three consecutive
`controlled` rows before ending an `in_transit` interval improved the second
Le Moyne review score from 78/108 to 80/108 labels. It reduced first-minute
state transitions from 54 to 46. The gains were mainly pass-near-defender cases;
the regressions were controlled frames around known ball-detection/close-control
edge cases, so the candidate needs a small visual review before promotion.

The visual review of that receiver-confirmation candidate marked 29 of 46
transitions correct and scored 72 of 92 reviewed frame labels. Remaining errors
were concentrated in:

- early-clip frames that should stay `controlled`;
- two short pass intervals that should stay `in_transit`;
- sideline/non-player detections becoming the nearest player;
- one remaining pass-near-defender case.

This points to player filtering near sidelines as the next highest-value
upstream fix before additional ball-state threshold tuning.

Initial sideline filtering tests found a pipeline issue: older filtered player
CSVs did not preserve `field_zone`, which made downstream association less
diagnostic. `src/43_filter_on_field_players.py` now supports legacy
`field_polygon` mask CSVs and preserves strict/tolerant zone labels in the
filtered output.

Le Moyne sideline experiments tested:

- v3 field polygon as a stricter playable-field mask;
- v3 strict polygon plus v4 convex tolerant polygon;
- strict-only controlled evidence;
- tolerant-edge receiver confirmation;
- removal of short tolerant-edge detections.

These removed some known sideline offenders locally, but none beat the current
receiver-confirmation candidate on the reviewed Le Moyne transition labels. The
best current production candidate remains the receiver-confirmed smoother
without the stricter sideline mask. A better sideline fix likely needs either
improved field segmentation near the far touchline or a player/referee/substitute
classifier rather than geometry-only filtering.

A follow-up player-eligibility review sampled 80 suspicious nearest-player
detections. The reviewer marked 58 as active players, 18 as substitutes/staff,
and 4 as referees. All substitute/staff nearest-player errors were in the
tolerant field zone, but all four referee nearest-player errors were in the
strict zone, so a field-boundary-only fix is not sufficient. Applying the
review as exact frame/player exclusions removed 22 nearest detections; parsing
review notes for ranked non-active boxes removed 92 total detections. Both
variants fixed local nearest-player evidence but reduced the receiver-confirmed
transition-review score from 72/92 to 68/92 because the smoother then
over-extended nearby `in_transit` intervals. Keep this as diagnostic evidence,
not as the next production candidate.

A later combined active-participant review added Army, Holy Cross, and Siena
examples. The reviewer labeled 138 of 148 nearest candidates as active players,
6 as substitutes/staff, and 4 as referees. In that multi-venue batch, every
non-active nearest candidate was in the tolerant zone, but 19 tolerant-zone
nearest candidates were still active players. Simple tolerant-zone rules either
catch all non-active cases with many false positives, or catch only the small
low-confidence substitute/staff cases and miss the Siena referee cases. When
the Le Moyne review is included, the simple tolerant-zone rules still miss the
strict-zone referee failure mode. The active-participant fix should therefore
be evaluated as a learned or track-aware eligibility layer, not promoted as a
single geometry threshold.

`src/59_build_active_participant_features.py` converts reviewed eligibility
manifests into a feature table for rule/model development. The first combined
feature table has 228 reviewed rows across Le Moyne, Army, Holy Cross, and
Siena. A transparent staff/referee-band rule catches 28 of 32 non-active
nearest candidates, but falsely flags 16 active players. That is useful for
analysis but too aggressive for production; the next tuning step should reduce
false positives before the layer is allowed to filter possession candidates.

A focused false-positive review found that 14 of those 16 were true false
positives, while 2 Le Moyne rows should actually be corrected to
substitute/staff. After correction, the reviewed feature set contains 34
non-active nearest candidates. The best zero-false-positive candidate catches
only 7 of 34 non-active cases; a more useful low-false-positive candidate
catches 17 of 34 with 3 reviewed active-player false positives. Treat this as
the next review candidate, not a production filter yet.

A follow-up in-chat review of the 3 active-player false positives from that
low-false-positive candidate confirmed all 3 green-boxed candidates as active
players. One Le Moyne image also showed a nearby `#3` box that was
`substitute_or_staff`, but that was not the reviewed green-box candidate. The
balanced low-false-positive rule therefore remains at 17 of 34 non-active cases
caught with 3 active-player false positives. The filter should still be applied
conservatively until reviewed on more games.

`src/60_score_active_participants.py` adds the first active-participant
advisory layer. It annotates association rows with:

- `active_participant_advisory_decision`
- `active_participant_advisory_reason`
- `active_participant_non_active_score`
- feature columns used by the current transparent rule bands

The advisory decisions are:

- `high_confidence_non_active`: safest reviewed rule band;
- `likely_non_active`: balanced low-false-positive rule band;
- `review_non_active_risk`: useful for review/model training, not filtering;
- `likely_active`: no current rule-band concern;
- `unknown`: association row could not be matched back to a player detection.

The script does not remove detections. `src/52_build_possession_chains.py` can
optionally ignore selected advisory decisions as control evidence via
`--exclude-active-participant-decisions`. On the Le Moyne first-minute test,
the normal advisory baseline produced 17 possession segments; excluding only
`high_confidence_non_active` candidates produced 14 segments; excluding both
`high_confidence_non_active` and `likely_non_active` also produced 14 segments.
This makes the conservative flag useful for comparison, while the balanced flag
still needs review before hard filtering.

Latest advisory outputs are in `outputs/active_participant_advisory/`.

An event-impact review on the Le Moyne first-minute candidate found that a
single-frame active-fallback repair around 34.10s fixed the visual state at one
frame but introduced wrong-owner risk because the interpolated ball position was
wrong and the fallback owner became a Colgate defender. The review marked 4 of
8 event-impact clips correct, 3 wrong-owner, and 1 wrong-state. Keep
`src/61_repair_ball_state_active_fallback.py` as diagnostic tooling for now,
not as part of the promoted candidate path. The safer current candidate is the
older receiver-confirm ball-state output plus high-confidence non-active
exclusion, without active-fallback repair.

The same event-impact review marked the start of `possession_0001` as
`wrong_state`: it was still a traveling pass, but the state machine started a
controlled possession from no current owner. `src/52_build_possession_chains.py`
now supports optional new-possession stability gates:

- `--start-max-speed`
- `--start-max-distance`

These gates apply only when starting possession from no current owner; confirmed
switches during an existing possession chain are unchanged. On the reviewed Le
Moyne first-minute candidate, `--start-max-speed 250 --start-max-distance 45`
blocks the bad start at frame 776 and removes the false first-minute controlled
segment.

An Army full-clip event-impact review tested the same tuned candidate on a
different venue. The review covered 14 clips: 7 possession starts and 7
possession ends. The labels were 4 `correct`, 4 `merge`, 3 `shift_start`, and 3
`wrong_state`.

The Army errors point to two issues beyond the Le Moyne pass-travel fix:

- controlled starts often begin too late because nearest-player identity
  flickers while the ball is already controlled;
- several reviewed moments happen around goals/free kicks/clearances where
  possession state alone cannot tell whether the game is live or stopped.

This argues against further tiny possession-threshold tweaks as the next main
step. The next production layer should either merge short same-phase controlled
fragments through detection flicker, or add an explicit game-state/dead-ball
classifier so stopped-play possession does not pollute event derivation.

`src/63_build_game_state_review.py` creates short review clips for that
game-state layer. The review labels are:

- `live`: normal active play; possession/event derivation should consider the
  moment.
- `dead_ball`: play is stopped and no restart action is currently happening.
- `restart_setup`: players are setting up for a restart, but the ball has not
  been put back into play.
- `restart_kick`: the restart touch/throw/kick is happening at the center
  moment.
- `goal_stoppage`: the moment is part of post-goal stoppage or kickoff reset.
- `unclear`: video context is not enough to decide.

The first Army game-state review package is seeded from the completed
event-impact review and saved in
`outputs/game_state_review/army_from_event_impact/`.

Example review outputs:

- `outputs/field_segmentation_strict_tolerant_evaluation/army_possession_chain_overlay_first60s.mp4`
- `outputs/lemoyne/lemoyne_short_clip_possession_chain_overlay_first60s.mp4`
- `outputs/lemoyne/ball_state_transition_review_first60s/pairs/`
- `outputs/field_segmentation_strict_tolerant_evaluation/army_ball_state_transition_review_first60s/pairs/`
- `outputs/possession_chain_tuning/lemoyne_smoothed_transition_review_first60s/pairs/`
- `outputs/possession_chain_tuning/lemoyne_receiver_confirm_transition_review_first60s/pairs/`

The threshold sweep in `outputs/possession_chain_tuning/threshold_sweep_summary.csv`
showed:

- `min_confirm_frames = 3` creates too many short possession segments.
- `min_confirm_frames = 8` or `12` misses too many legitimate possessions.
- `min_confirm_frames = 5` remains the best current default.
- A `2.0` second carry-forward timeout prevents indefinite sticky ownership
  while preserving normal pass/travel continuity.

## Initial Runs

With the tuned V1 settings:

| Clip | Association rows | Raw controlled nearest-player switches | Confirmed possession segments |
| --- | ---: | ---: | ---: |
| Army short clip | 4,118 | 2,321 | 17 |
| Le Moyne short clip | 4,418 | 2,427 | 18 |

These counts are not final accuracy metrics. They show that the state-machine
layer suppresses frame-to-frame nearest-player flicker and produces a smaller
set of possession intervals suitable for video review.
