# Manual Next Steps

This file tracks the work that needs human review while Codex works on the
pipeline around it.

## Current Queue

1. While the Roboflow ball model trains, review the LeMoyne combined
   non-active participant filter overlay:
   - Review:
     `outputs/active_participant_review/lemoyne_tracked_min15_non_active_review/ranked_note_extraction/lemoyne_combined_filtered_min15_overlay_first60s.mp4`
   - Compare to:
     `outputs/active_participant_review/lemoyne_tracked_min15_non_active_review/lemoyne_track_filtered_min15_overlay_first60s.mp4`
   - Focus question: did removing referee track `114`, plus tracks `145` and
     `126`, reduce sideline/referee control mistakes without making real
     possession too conservative?
2. When Roboflow training finishes, provide the YOLOv8 `best.pt` path.
   - If exported YOLO labels are not ready, run model-only overlay mode first:
     `.venv/bin/python src/82_run_post_roboflow_handoff.py --model path/to/best.pt --full-clips lemoyne,army`
   - If labels are available, run full validation/evaluation mode:
     `.venv/bin/python src/82_run_post_roboflow_handoff.py --model path/to/best.pt --labels-dir path/to/labels --full-clips lemoyne,army`

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

- The 65-image LeMoyne non-active review is complete. The conservative combined
  filter currently excludes tracks `145` (referee), `114` (referee), and `126`
  (substitute/staff). Do not auto-remove mixed/sparse tracks such as `103`,
  `130`, `117`, `177`, or `23` without more review.
- The Roboflow annotation batch is complete and training is in progress. After
  `best.pt` is available, run `src/82_run_post_roboflow_handoff.py`.

## Codex-Side Queue

1. Preserve or port the local ranked-note extraction and merged eligibility
   summary scripts if they are needed in this checkout:
   `src/84_extract_ranked_eligibility_notes.py` and
   `src/85_merge_eligibility_track_summaries.py`.
2. Run the post-Roboflow handoff once `best.pt` is available, starting with
   model-only overlay mode if labels are not ready.

## Notes

- The non-active participant review helps prevent sideline substitutes, staff,
  and referees from stealing possession ownership from active players.
- The ball batch is the bigger upstream fix. It should reduce bad possession
  states caused by missing balls, wrong balls, and suspicious ball jumps.
