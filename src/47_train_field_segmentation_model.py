#!/usr/bin/env python3

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO segmentation dataset YAML")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="playable_field")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=args.project,
        name=args.name,
        patience=25,
        degrees=2.0,
        perspective=0.0005,
        scale=0.35,
        fliplr=0.5,
        plots=True,
    )


if __name__ == "__main__":
    main()
