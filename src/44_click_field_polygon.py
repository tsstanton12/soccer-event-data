import argparse
import cv2
from pathlib import Path

points = []

def click_event(event, x, y, flags, param):
    global points

    frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked: {x},{y}")

        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0, 255, 255), 2)

        cv2.imshow("Click field polygon points", frame)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame}")

    display = frame.copy()

    print("Click polygon points around the playable field area.")
    print("Press any key when done.")
    print("Use points in clockwise or counter-clockwise order.")

    cv2.imshow("Click field polygon points", display)
    cv2.setMouseCallback("Click field polygon points", click_event, display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("Need at least 3 points.")
        return

    polygon_string = ";".join([f"{x},{y}" for x, y in points])

    print("\nFIELD POLYGON:")
    print(polygon_string)

    print("\nUse like this:")
    print(f'--field_polygon "{polygon_string}"')

if __name__ == "__main__":
    main()