import cv2
import pandas as pd
from pathlib import Path

VIDEO_PATH = "videos/match_clip_3.mp4"
BALL_CSV = "outputs/test_clip_ball_detections.csv"
OUTPUT_DIR = Path("review_training_frames")

OUTPUT_DIR.mkdir(exist_ok=True)

# Criteria you can adjust
LOW_CONF_MIN = 0.10
LOW_CONF_MAX = 0.35
MAX_FRAMES_TO_EXPORT = 150

df = pd.read_csv(BALL_CSV)

# Pick frames where model is unsure
low_conf = df[
    (df["confidence"] >= LOW_CONF_MIN) &
    (df["confidence"] <= LOW_CONF_MAX)
].copy()

# Pick frames with lots of detections — often false-positive chaos
detections_per_frame = df.groupby("frame").size().reset_index(name="count")
busy_frames = detections_per_frame[detections_per_frame["count"] >= 3]["frame"]

candidate_frames = set(low_conf["frame"].tolist()) | set(busy_frames.tolist())
candidate_frames = sorted(candidate_frames)

# Limit export count
candidate_frames = candidate_frames[:MAX_FRAMES_TO_EXPORT]

cap = cv2.VideoCapture(VIDEO_PATH)

saved = 0

print(f"Exporting {len(candidate_frames)} review frames...")

for frame_num in candidate_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))

    ret, frame = cap.read()
    if not ret:
        continue

    output_path = OUTPUT_DIR / f"review_frame_{int(frame_num):06d}.jpg"
    cv2.imwrite(str(output_path), frame)
    saved += 1

cap.release()

print(f"Done! Saved {saved} frames to {OUTPUT_DIR}")