import cv2
import pandas as pd

VIDEO_PATH = "videos/siena_1st.mp4"
BALL_CSV = "outputs/conservative_ball_path.csv"
OUTPUT_VIDEO_PATH = "outputs/conservative_ball_path_preview.mp4"

ball = pd.read_csv(BALL_CSV)
ball = ball.sort_values("frame").copy()

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

ball_by_frame = {
    frame: group for frame, group in ball.groupby("frame")
}

recent_points = []
TRAIL_LENGTH = 30

frame_idx = 0

print("Creating ball trail preview...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in ball_by_frame:
        detections = ball_by_frame[frame_idx]

        for _, row in detections.iterrows():
            cx = int(row["center_x"])
            cy = int(row["center_y"])
            conf = row["confidence"]

            recent_points.append((cx, cy))

            if len(recent_points) > TRAIL_LENGTH:
                recent_points = recent_points[-TRAIL_LENGTH:]

            cv2.circle(frame, (cx, cy), 9, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ball {conf:.2f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    # Draw actual recent trail
    for i in range(1, len(recent_points)):
        cv2.line(
            frame,
            recent_points[i - 1],
            recent_points[i],
            (255, 0, 0),
            3
        )

    cv2.putText(
        frame,
        "Filtered Ball Trail",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        3,
    )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Done! Saved trail preview to {OUTPUT_VIDEO_PATH}")