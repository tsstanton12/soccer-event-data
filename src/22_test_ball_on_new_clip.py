from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# CHANGE THESE FOR EACH TEST CLIP
VIDEO_PATH = "videos/match_clip.mov"
MODEL_PATH = "runs/detect/train-11/weights/best.pt"
OUTPUT_PATH = "outputs/test_clip_ball_detections.csv"

CONFIDENCE_THRESHOLD = 0.20

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

rows = []
frame_idx = 0

Path("outputs").mkdir(exist_ok=True)

print(f"Running ball detector on {VIDEO_PATH}...")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        if label.lower() != "ball":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        rows.append({
            "frame": frame_idx,
            "time_seconds": frame_idx / fps,
            "object_type": "ball",
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        })

    frame_idx += 1

cap.release()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone! Saved {len(df)} ball detections to {OUTPUT_PATH}")