import cv2
import json
from pathlib import Path

VIDEO_PATH = "videos/siena_1st.mp4"
OUTPUT_PATH = "outputs/field_polygon.json"

Path("outputs").mkdir(exist_ok=True)

# Choose which frame to use for clicking the field boundary
FRAME_NUMBER = 1500

points = []

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

    for point in points:
        cv2.circle(display, point, 8, (0, 0, 255), -1)

    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], (255, 0, 0), 3)

    if len(points) > 2:
        cv2.line(display, points[-1], points[0], (255, 0, 0), 2)

    cv2.putText(
        display,
        "Click field boundary points. Press S to save, U to undo, Q to quit.",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()

redraw()

cv2.namedWindow("Create Field Mask", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Create Field Mask", mouse_callback)

print("Instructions:")
print("- Click around the playable field boundary.")
print("- Use 4-8 points.")
print("- Press U to undo last point.")
print("- Press S to save.")
print("- Press Q to quit without saving.")

while True:
    cv2.imshow("Create Field Mask", display)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u"):
        if points:
            points.pop()
            redraw()

    elif key == ord("s"):
        if len(points) < 3:
            print("Need at least 3 points to save a polygon.")
            continue

        polygon_data = {
            "video_path": VIDEO_PATH,
            "frame_number": FRAME_NUMBER,
            "points": points,
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(polygon_data, f, indent=2)

        print(f"Saved field polygon to {OUTPUT_PATH}")
        break

    elif key == ord("q"):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()