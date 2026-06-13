import cv2
import pandas as pd

VIDEO_PATH = "videos/match_clip.mov"
BALL_CSV = "outputs/conservative_ball_path.csv"
PROGRESSION_CSV = "outputs/ball_progressions.csv"
OUTPUT_VIDEO_PATH = "outputs/ball_progressions_preview.mp4"

ball = pd.read_csv(BALL_CSV)
progressions = pd.read_csv(PROGRESSION_CSV)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

ball_by_frame = {
    frame: group for frame, group in ball.groupby("frame")
}

progression_rows = progressions.to_dict("records")

frame_idx = 0

print("Creating ball progression preview...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw current tracked ball
    if frame_idx in ball_by_frame:
        detections = ball_by_frame[frame_idx]

        for _, row in detections.iterrows():
            cx = int(row["center_x"])
            cy = int(row["center_y"])
            conf = row["confidence"]

            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ball {conf:.2f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    # Draw active progression
    for prog in progression_rows:
        start_frame = int(prog["start_frame"])
        end_frame = int(prog["end_frame"])

        if start_frame <= frame_idx <= end_frame:
            start_point = (int(prog["start_x"]), int(prog["start_y"]))
            end_point = (int(prog["end_x"]), int(prog["end_y"]))

            cv2.line(frame, start_point, end_point, (255, 0, 0), 4)
            cv2.circle(frame, start_point, 10, (255, 0, 0), 2)
            cv2.circle(frame, end_point, 10, (255, 0, 0), 2)

            label = (
                f"Progression {int(prog['sequence_id'])} | "
                f"{prog['duration_seconds']:.1f}s | "
                f"{prog['pixel_distance']:.0f}px"
            )

            cv2.putText(
                frame,
                label,
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

print(f"Done! Saved preview to {OUTPUT_VIDEO_PATH}")