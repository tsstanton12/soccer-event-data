import cv2
import json
from pathlib import Path

VIDEO_PATH = "videos/match_clip.mov"
OUTPUT_PATH = "outputs/field_calibration_points.json"

# Change this to a frame where field lines are visible
FRAME_NUMBER = 1200

Path("outputs").mkdir(exist_ok=True)

# Known field landmarks in yards
# Field model: x = 0 to 120, y = 0 to 75
LANDMARKS = [
    ("center_spot", 60, 37.5),
    ("top_touchline_halfway", 60, 0),
    ("bottom_touchline_halfway", 60, 75),
    ("left_18_top_corner", 18, 15.5),
    ("left_18_bottom_corner", 18, 59.5),
]

points = []
current_landmark_index = 0

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read video frame. Check VIDEO_PATH and FRAME_NUMBER.")

display = frame.copy()


def redraw():
    global display

    display = frame.copy()

    for item in points:
        px, py = item["pixel"]
        name = item["name"]

        cv2.circle(display, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            name,
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if current_landmark_index < len(LANDMARKS):
        name, field_x, field_y = LANDMARKS[current_landmark_index]
        instruction = f"Click: {name}  field=({field_x}, {field_y})"
    else:
        instruction = "All landmarks clicked. Press S to save."

    cv2.putText(
        display,
        instruction,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "Press U to undo, S to save, Q to quit.",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )


def mouse_callback(event, x, y, flags, param):
    global current_landmark_index

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_landmark_index >= len(LANDMARKS):
            return

        name, field_x, field_y = LANDMARKS[current_landmark_index]

        points.append({
            "name": name,
            "pixel": [int(x), int(y)],
            "field": [float(field_x), float(field_y)],
        })

        current_landmark_index += 1
        redraw()


redraw()

cv2.namedWindow("Create Field Calibration", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Create Field Calibration", mouse_callback)

print("Instructions:")
print("- Click each requested field landmark in order.")
print("- If the landmark is NOT visible, press U only if needed, then quit and choose a better FRAME_NUMBER.")
print("- Press U to undo last point.")
print("- Press S to save.")
print("- Press Q to quit.")

while True:
    cv2.imshow("Create Field Calibration", display)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u"):
        if points:
            points.pop()
            current_landmark_index = max(0, current_landmark_index - 1)
            redraw()

    elif key == ord("s"):
        if len(points) < 4:
            print("Need at least 4 points to save calibration.")
            continue

        calibration_data = {
            "video_path": VIDEO_PATH,
            "frame_number": FRAME_NUMBER,
            "field_length_yards": 120,
            "field_width_yards": 75,
            "points": points,
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(calibration_data, f, indent=2)

        print(f"Saved calibration points to {OUTPUT_PATH}")
        break

    elif key == ord("q"):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()