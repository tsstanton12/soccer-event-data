from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

VIDEO_PATH = "videos/match_clip.mov"
OUTPUT_PATH = "outputs/ball_detections.csv"

model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

rows = []
frame_idx = 0

Path("outputs").mkdir(exist_ok=True)

print("Starting ball detection on every frame...")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        if label == "sports ball" and confidence >= 0.05:
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

print(f"Done! Saved {len(df)} ball detections to {OUTPUT_PATH}")