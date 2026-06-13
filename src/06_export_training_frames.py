import cv2
from pathlib import Path

VIDEO_PATH = "videos/albany_clip.mp4"
OUTPUT_DIR = Path("more_training_frames")

OUTPUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
saved = 0

# Save one frame every 1 seconds
fps = cap.get(cv2.CAP_PROP_FPS)
save_every = int(fps * 1)

print("Exporting training frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % save_every == 0:
        output_path = OUTPUT_DIR / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved += 1

    frame_idx += 1

cap.release()

print(f"Done! Saved {saved} images to {OUTPUT_DIR}")