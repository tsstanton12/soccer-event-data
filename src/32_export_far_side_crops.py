import cv2
from pathlib import Path

VIDEO_PATH = "videos/siena_1st.mp4"
OUTPUT_DIR = Path("far_side_training_crops")

OUTPUT_DIR.mkdir(exist_ok=True)

# Adjust this depending on where the far side is in your video.
# This default crops the upper half / far side of a 1920x1080 video.
CROP_X1 = 0
CROP_Y1 = 150
CROP_X2 = 1920
CROP_Y2 = 600

# Export one crop every 2 seconds
SECONDS_BETWEEN_EXPORTS = 2

MAX_CROPS = 500

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_step = int(fps * SECONDS_BETWEEN_EXPORTS)

saved = 0
frame_idx = 0

print("Exporting far-side crops...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_step == 0:
        crop = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

        output_path = OUTPUT_DIR / f"far_side_crop_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), crop)

        saved += 1

        if saved >= MAX_CROPS:
            break

    frame_idx += 1

cap.release()

print(f"Done! Saved {saved} crops to {OUTPUT_DIR}")