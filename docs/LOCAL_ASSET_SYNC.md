# Local Asset Sync

The code and lightweight review data belong in Git. Large working assets should
be available to the machine running the pipeline, but should not be committed to
normal Git history.

## Keep Out Of Git

These paths and file types are intentionally ignored:

- `videos/`
- `models/`
- `outputs/`
- `runs/`
- `roboflow_zips/`
- `*.pt`, `*.pth`, `*.onnx`, `*.zip`

That includes ball model weights such as `best_june_17.pt`.

## What A Cloud Agent Needs To Run The Full Pipeline

For code-only work, the repository is usually enough. For full evaluation or
overlay generation, the cloud workspace also needs the relevant local assets:

- input videos, for example `videos/lemoyne_short_clip.mp4` and
  `videos/army_short_clip.mp4`;
- model weights, for example `models/best_june_17.pt` or another path supplied
  to `--model`;
- tracked player CSVs and track-stability summaries under `outputs/`;
- Roboflow hard-frame images and, when available, exported YOLO labels;
- review outputs that are being compared visually.

If those files are only on the local Mac, sync or upload the needed subset before
asking a cloud agent to run the pipeline. Do not add the whole project folder to
Git unless the files are source code, docs, notebooks, small manifests, or small
review labels that should be versioned.

## Current Latest Ball Model

The latest reported ball model artifact is:

```text
best_june_17.pt
```

Suggested local location:

```text
models/best_june_17.pt
```

Example model-only handoff command:

```sh
.venv/bin/python src/82_run_post_roboflow_handoff.py \
  --model models/best_june_17.pt \
  --full-clips lemoyne,army
```
