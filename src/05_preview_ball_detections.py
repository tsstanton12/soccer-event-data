import cv2
import pandas as pd

VIDEO_PATH = "videos/match_clip_2.mp4"
CSV_PATH = "outputs/ball_detections.csv"
OUTPUT_VIDEO_PATH = "outputs/ball_detection_preview.mp4"

df = pd.read_csv(CSV_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

detections_by_frame = {
    frame: group for frame, group in df.groupby("frame")
}

frame_idx = 0

print("Creating ball preview video...")

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
            conf = row["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(
                frame,
                (int(row["center_x"]), int(row["center_y"])),
                8,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                f"ball {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Done! Saved preview video to {OUTPUT_VIDEO_PATH}")