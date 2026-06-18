# Event Data Project Roadmap

This roadmap resets the project around functional layers rather than around a
single possession/event script. The main lesson so far is that event derivation
is downstream of several perception tasks. When those upstream layers are only
partly reliable, possession and events become hard to interpret.

## Current Strategy

Work from perception toward events:

1. Make ball detection/tracking measurable across venues.
2. Make player detection/tracking stable enough for possession ownership.
3. Filter non-active people before they can become possession owners.
4. Classify possession state and possession owner.
5. Add team identity.
6. Add live/dead/restart game-state filtering.
7. Derive simple events.
8. Expand to richer event types.

## Functional Status

| Area | Status | Current Estimate | Main Blocker |
| --- | --- | ---: | --- |
| Ball detection | Active reset | 45% | Bad/missing ball detections in hard broadcast conditions. |
| Ball tracking | Provisional | 50% | Needs better bad-detection rejection and evaluation after retraining. |
| Player detection | Usable baseline | 55% | Generic person detector catches non-players and misses some edge cases. |
| Player identification | Early | 20% | No true re-identification yet. |
| Player tracking | Early/provisional | 35% | Simple track IDs need review for swaps/fragments. |
| Team identification | Paused | 20% | Jersey clustering depends on correct possession owner/track context. |
| Non-active filtering | Provisional | 35% | Substitutes, assistant referees, and sideline figures still leak through. |
| Playable area detection | Strongest layer | 70% | Conservative edge misses remain but strict/tolerant approach is useful. |
| Field zone mapping | Early | 20% | Need stable camera-to-field mapping before x/y event locations. |
| Game state identification | Review workflow exists | 35% | Needs classifier/rules after more reviewed examples. |
| Possession state | Improving | 45% | Sensitive to ball quality and non-active people. |
| Possession ownership | Provisional | 30% | Needs stable ball + track IDs + non-active filtering. |
| Possession chain tuning | Useful but downstream | 50% | Should not compensate for bad ball detections indefinitely. |
| Simple event recognition | Early | 10% | Pass candidates exist, but upstream reliability is not enough yet. |
| Complex event recognition | Not started | 0-5% | Wait until simple events are trustworthy. |

The percentages are rough engineering estimates, not formal benchmarks. They
exist to help choose the next task.

## Immediate Work

### 1. Ball Detection Retraining And Evaluation

Status: annotation complete; new model training/export pending.

A 750-frame multi-venue hard-case batch was generated for Roboflow:

- `outputs/ball_detection_review/multi_venue_750/clean_frames/`
- `outputs/ball_detection_review/multi_venue_750/annotated_frames/`
- `outputs/ball_detection_review/multi_venue_750/ball_detection_review_manifest.csv`

The batch covers Army, LeMoyne, Siena, Holy Cross, and Albany. It focuses on
low-confidence detections, detection gaps, gap-middle frames, and suspicious
ball jumps.

Manual work:

- Wait for the Colab YOLOv8 training run to finish and provide the exported
  `best.pt` path.
- Export YOLO labels if available; labels are useful for validation/evaluation
  but not required for the first model-only full-clip overlay rerun.

Automated work:

- Evaluate the new model on this same batch.
- Compare recall on visible-ball frames, false positives on null frames, and
  localization quality by venue/reason.

### 2. Player Track Review

Status: built, needs review.

Simple persistent track outputs:

- `outputs/player_tracking/army_players_on_field_tracked.csv`
- `outputs/player_tracking/lemoyne_players_on_field_tracked.csv`
- `outputs/player_tracking/army_track_overlay_first60s.mp4`
- `outputs/player_tracking/lemoyne_track_overlay_first60s.mp4`

Manual work:

- Review whether track IDs stay attached to the same physical player.
- Note frequent swaps/fragments.

Automated work:

- Tune simple tracker thresholds.
- Replace frame-local IDs in tuned association rows.

### 3. Non-Active Participant Filtering

Status: LeMoyne 65-image review complete; combined filter needs visual review.

LeMoyne is close to usable possession flow, but remaining errors often involve
substitutes, staff, or assistant referees. The conservative combined filter now
excludes tracks `145` and `114` as referees and track `126` as
substitute/staff. It needs visual review before it is treated as promoted
behavior.

Manual work:

- Compare the combined-filter first-minute overlay against the earlier
  track-filtered overlay and verify whether removing track `114` reduces
  sideline/referee ownership errors without hiding real possessions.

Automated work:

- Train/evaluate an active-participant layer or stronger rules.

## Order Of Operations

Do not push event derivation forward until the first two items improve:

1. Finish Roboflow training/export for the next ball model.
2. Run the hard-batch ball evaluation.
3. Rerun ball paths on key clips.
4. Rebuild association and min-15 possession chains.
5. Review remaining possession errors.
6. Add track-stability gating so unstable tracks cannot dominate possession
   ownership, especially in the Army clip.
7. Improve non-active filtering.
8. Revisit team identity.
9. Resume simple pass/carry/restart event derivation.

This order keeps us from tuning possession around errors caused by the ball
model.
