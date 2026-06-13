# Soccer Event Data

An experimental pipeline for turning broadcast soccer video into structured
event data. The current work covers player and ball detection, ball-path
interpolation, playable-field segmentation, on-field player filtering,
ball-player association, and early possession/event labeling.

## Project Map

- `src/`: numbered computer-vision and event-processing scripts
- `notebooks/`: model-training notebooks
- `data/field_segmentation/`: lightweight field-segmentation documentation and
  annotation manifest
- `soccer-xT-project/`: exploratory event-schema and expected-threat work
- `tools/possession-lab/`: browser tool for reviewing possession labels and
  completed-pass candidates
- `docs/PROJECT_CONTEXT.md`: current decisions, results, and roadmap

Large local assets such as videos, model weights, training images, generated
outputs, and experiment runs are intentionally excluded from Git.

## Possession Lab

```sh
cd tools/possession-lab
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

