from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

# Resume from latest checkpoint
MODEL_PATH = "runs/detect/train-12/weights/last.pt"

# Your dataset yaml
DATA_YAML = "soccer-ball-detector.v10i.yolov8/data.yaml"

# Training settings
TOTAL_EPOCHS = 50
IMAGE_SIZE = 1280
BATCH_SIZE = 4

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# TRAIN / RESUME TRAINING
# ============================================================

model.train(
    data=DATA_YAML,
    epochs=TOTAL_EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device="cpu",
    workers=0,

    # Resume previous run
    resume=True,

    # Save/checkpoint settings
    save=True,
    save_period=1,

    # Helpful training options
    plots=True,
    verbose=True,
)

print("\nTraining complete.")
