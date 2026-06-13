from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse


def save_frame(frame, out_dir, video_stem, frame_num):
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{video_stem}_frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filename), frame)


def harvest_frames(video_path, model_path, venue, output_root, conf_threshold=0.15):
    video_path = Path(video_path)
    output_root = Path(output_root)
    model = YOLO(model_path)

    tiny_count = 0
    low_conf_count = 0
    negative_count = 0
    review_count = 0
    tracking_failure_count = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_stem = video_path.stem
    frame_num = 0
    no_detection_streak = 0
    previous_detection_frame = None

    print(f"Harvesting frames from: {video_path}")
    print(f"Venue: {venue}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Process every 10th frame to avoid too many near-duplicates
        if frame_num % 10 != 0:
            continue

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        venue_dir = output_root / venue

        if boxes is None or len(boxes) == 0:
            no_detection_streak += 1

            if no_detection_streak % 5 == 0:
                save_frame(frame, venue_dir / "negative_frames", video_stem, frame_num)
                negative_count += 1

            continue

        no_detection_streak = 0

        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        conf = float(best_box.conf[0])
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()

        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h

        if area < 150:
            save_frame(frame, venue_dir / "tiny_ball", video_stem, frame_num)
            tiny_count += 1

        if conf < 0.40:
            save_frame(frame, venue_dir / "low_confidence", video_stem, frame_num)
            low_conf_count += 1

        if frame_num % 100 == 0:
            save_frame(frame, venue_dir / "review_all", video_stem, frame_num)
            review_count += 1

        if previous_detection_frame is not None:
            gap = frame_num - previous_detection_frame
            if gap > 75:
                save_frame(frame, venue_dir / "tracking_failures", video_stem, frame_num)
                tracking_failure_count += 1

        previous_detection_frame = frame_num

    cap.release()

    print("\nHarvest Summary")
    print("----------------")
    print(f"Tiny ball frames: {tiny_count}")
    print(f"Low confidence frames: {low_conf_count}")
    print(f"Negative frames: {negative_count}")
    print(f"Review frames: {review_count}")
    print(f"Tracking failure frames: {tracking_failure_count}")
    print("Done harvesting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--venue", required=True, help="Venue name, e.g. patriot_league/colgate")
    parser.add_argument("--output", default="data/harvested_frames", help="Output folder")

    args = parser.parse_args()

    harvest_frames(
        video_path=args.video,
        model_path=args.model,
        venue=args.venue,
        output_root=args.output
    )