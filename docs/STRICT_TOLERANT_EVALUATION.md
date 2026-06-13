# Strict/Tolerant Field Polygon Evaluation

## Scope

Evaluated the V2 playable-field model on the previously reviewed Army and
Siena clips using the current downstream settings:

- segmentation inset: `0px`
- tolerant polygon expansion: `15px`
- sample interval: 300 frames
- reviewed frames: 48

The five previously labeled `too_tight` frames were:

- Army: frames 3600 and 7800
- Siena: frames 0, 300, and 2100

## Results

- Field detected in all 48 sampled frames.
- No low-confidence predictions.
- Removing the old 3px inset increased strict-polygon area by an average of
  0.68 percentage points.
- At the five previously too-tight frames, strict-polygon area increased by
  0.60 to 0.76 percentage points.

Army has player detections available for downstream evaluation:

- total player detections: 285,327
- strict detections: 267,299
- tolerant-only detections retained: 5,120
- off-field detections excluded: 12,908

At both known Army failure frames:

- 20 player detections were inside the strict polygon;
- 0 were tolerant-only;
- 1 extreme-right sideline detection remained off-field.

Visual review indicates the excluded extreme-right detections are sideline
figures rather than players who should be eligible for possession. Ball-player
association around both frames selected strict-zone players.

Siena does not currently have player-detection CSVs, so its three frames were
evaluated visually. The strict boundary remains conservative at empty field
edges, while the tolerant boundary covers the small cutoff. No player or ball
is lost in the affected space.

## Decision

Keep the current production settings:

- `segmentation_inset_px = 0`
- `tolerant_margin_px = 15`

Continue preferring strict players for ball association and only permit
tolerant-only players when no strict player is within association range.

Do not widen the learned segmentation annotations or increase the tolerant
margin based on these cases.

