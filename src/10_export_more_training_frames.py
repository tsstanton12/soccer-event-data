import cv2
from pathlib import Path

VIDEO_PATH = "videos/lafayette_clip.mp4"
OUTPUT_DIR = Path("future_training_images")

OUTPUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0
saved = 0

# Save one frame every 1 second
save_every = int(fps * 1)

print("Exporting more training frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % save_every == 0:
        output_path = OUTPUT_DIR / f"clip4_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved += 1

    frame_idx += 1

cap.release()

print(f"Done! Saved {saved} images to {OUTPUT_DIR}")