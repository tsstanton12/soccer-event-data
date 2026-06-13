from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Detect players/persons in every frame of a soccer video."
    )

    parser.add_argument(
        "--venue",
        required=True,
        help="Venue/output name, e.g. army or lemoyne"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file"
    )

    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="YOLO model path. Default: yolov8x.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = args.model

    output_dir = Path("outputs") / args.venue
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{video_path.stem}_player_detections.csv"

    print("PLAYER DETECTION")
    print("----------------")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Venue: {args.venue}")
    print(f"Output CSV: {output_csv}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frame_idx = 0

    print("\nRunning player detection on every frame...")

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]

        player_num = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            if label != "person":
                continue

            player_num += 1

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append({
                "venue": args.venue,
                "video": video_path.name,
                "frame": frame_idx,
                "time_seconds": frame_idx / fps,
                "object_type": "player",
                "player_id": player_num,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            })

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\nPLAYER DETECTION COMPLETE")
    print("-------------------------")
    print(f"Saved rows: {len(df)}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()