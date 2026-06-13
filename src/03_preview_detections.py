import cv2
import pandas as pd
from pathlib import Path

VIDEO_PATH = "videos/match_clip.mov"
CSV_PATH = "outputs/detections.csv"
OUTPUT_VIDEO_PATH = "outputs/detection_preview.mp4"

df = pd.read_csv(CSV_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

detections_by_frame = {
    frame: group for frame, group in df.groupby("frame")
}

frame_idx = 0

print("Creating preview video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in detections_by_frame:
        detections = detections_by_frame[frame_idx]

        for _, row in detections.iterrows():
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            label = row["object_type"]
            conf = row["confidence"]

            if label == "person":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Done! Saved preview video to {OUTPUT_VIDEO_PATH}")