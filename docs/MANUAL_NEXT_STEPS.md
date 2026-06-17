# Manual Next Steps

This file tracks the work that needs human review while Codex works on the
pipeline around it.

## Current Queue

1. Complete the LeMoyne tracked non-active participant review.
   - Open:
     `outputs/active_participant_review/lemoyne_tracked_min15_non_active_review/player_eligibility_review_order.html`
   - Fill:
     `outputs/active_participant_review/lemoyne_tracked_min15_non_active_review/player_eligibility_review_order.csv`
   - Label the `#1 nearest box` in each image as one of:
     `active_player`, `referee`, `substitute_or_staff`, `off_field_player`, or
     `unclear`.
2. Continue annotating the 750-frame ball-detection batch in Roboflow.
   - Upload/annotate:
     `outputs/ball_detection_review/multi_venue_750/clean_frames/`
   - Reference images only:
     `outputs/ball_detection_review/multi_venue_750/annotated_frames/`
   - Use the single label `ball`.
   - If the ball is not visible, leave the image as a null/no-object frame.

## Optional Review

- If there is extra time after the two current queue items, compare the
  stability-gated first-minute possession overlays:
  - `outputs/player_tracking/association_rerun/stability_gated/army_tuned_tracked_min15_stability_gated_overlay_first60s.mp4`
  - `outputs/player_tracking/association_rerun/stability_gated/lemoyne_tuned_tracked_min15_stability_gated_overlay_first60s.mp4`
- Focus question: does stability gating reduce bad ownership switches without
  hiding real quick possessions?
- Short event-impact review pages are also available:
  - `outputs/player_tracking/association_rerun/stability_gated/event_impact_review/army_first60/event_impact_review_order.html`
  - `outputs/player_tracking/association_rerun/stability_gated/event_impact_review/lemoyne_first60/event_impact_review_order.html`

## Waiting On Completion

- After the 65-image non-active review is complete, summarize reviewed player
  IDs with `src/76_summarize_player_eligibility_reviews.py`, then apply
  reviewed non-active tracks with `src/78_apply_track_eligibility_reviews.py`.
- After the Roboflow batch is complete, train/export the next ball model and run
  the hard-frame evaluation workflow in `docs/BALL_MODEL_EVALUATION.md`.
  Start by validating the Roboflow export with
  `src/80_validate_roboflow_ball_export.py`.

## Codex-Side Queue

1. Review the stability-gated possession overlays and decide whether the gate
   improves Army/LeMoyne behavior.
2. Clean up the full-video ball-model rerun path for the next Roboflow model.

## Notes

- The non-active participant review helps prevent sideline substitutes, staff,
  and referees from stealing possession ownership from active players.
- The ball batch is the bigger upstream fix. It should reduce bad possession
  states caused by missing balls, wrong balls, and suspicious ball jumps.
