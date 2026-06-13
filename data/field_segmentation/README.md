# Playable-field segmentation dataset

## Label definition

Create one polygon class named `playable_field`.

- Label only the visible portion of the regulation soccer playing surface.
- Put polygon edges on the playable side of touchlines and endlines.
- Exclude green aprons, tracks, benches, technical areas, stands, and warm-up areas.
- Include worn, muddy, shadowed, overexposed, or partially obscured playable grass.
- Follow the inferred straight boundary through players, referees, goals, and other occlusions.
- Stop at the image edge when a field boundary is outside the camera frame.
- Ignore interior soccer lines and markings from other sports.

## Dataset design

The important source of diversity is venues, not nearby frames from one match.

- Annotate at least 30-50 distinct frames per venue to begin.
- Include daylight, night, shadows, glare, rain, natural grass, and artificial turf.
- Include wide, zoomed, panning, corner, midfield, and penalty-area views.
- Add hard negatives such as replays, crowd shots, and frames with little visible field.
- Split train/validation/test by entire venue or video. Never randomly split adjacent frames.
- Keep at least two venues completely unseen until final testing.

## Suggested first milestone

1. Annotate 400-600 frames across at least 10 venues.
2. Train a small YOLO segmentation model.
3. Test on two fully held-out venues.
4. Add the worst failures back into training and repeat.

The model's job is semantic: distinguish playable field from visually similar areas outside
the painted boundaries. Script 45 then converts its mask into a smooth convex polygon and
can apply temporal smoothing before player filtering.

## Commands

Harvest a diverse annotation set:

```bash
python src/46_harvest_field_segmentation_frames.py \
  --inputs data/raw_videos videos \
  --output_dir data/field_segmentation/images_to_annotate \
  --samples_per_video 40
```

After exporting a YOLO segmentation dataset, train it:

```bash
python src/47_train_field_segmentation_model.py \
  --data path/to/field-segmentation/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 100 \
  --imgsz 960
```

Use the trained model in the field-mask pipeline:

```bash
python src/45_auto_field_mask.py \
  --video videos/lemoyne_short_clip.mp4 \
  --segmentation_model runs/segment/playable_field/weights/best.pt \
  --output_csv outputs/lemoyne/lemoyne_field_masks.csv \
  --sample_every 30 \
  --debug_dir outputs/lemoyne/field_mask_debug
```
