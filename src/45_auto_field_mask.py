#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def polygon_string(poly):
    pts = poly.reshape(-1, 2)
    return ";".join([f"{int(x)},{int(y)}" for x, y in pts])


def simplify_polygon(poly, max_points=6):
    perimeter = cv2.arcLength(poly, True)

    for eps in np.linspace(0.002, 0.12, 60):
        approx = cv2.approxPolyDP(poly, eps * perimeter, True)
        if 4 <= len(approx) <= max_points:
            return approx

    return cv2.boxPoints(cv2.minAreaRect(poly)).astype(np.int32).reshape(-1, 1, 2)


def build_green_mask(frame, lower_green, upper_green):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_green, dtype=np.uint8),
        np.array(upper_green, dtype=np.uint8),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))

    return mask


def build_row_envelope(
    mask,
    min_row_green_frac=0.12,
    left_percentile=3.0,
    right_percentile=97.0,
):
    h, w = mask.shape

    left_points = []
    right_points = []

    # Work in horizontal bands to avoid jagged blobs.
    band_h = max(8, h // 80)

    for y0 in range(0, h, band_h):
        y1 = min(h, y0 + band_h)
        band = mask[y0:y1, :]

        cols = np.where(band.max(axis=0) > 0)[0]

        if len(cols) < w * min_row_green_frac:
            continue

        x_left = int(np.percentile(cols, left_percentile))
        x_right = int(np.percentile(cols, right_percentile))
        y_mid = int((y0 + y1) / 2)

        left_points.append([x_left, y_mid])
        right_points.append([x_right, y_mid])

    if len(left_points) < 8 or len(right_points) < 8:
        return None

    pts = np.array(left_points + right_points[::-1], dtype=np.int32).reshape(-1, 1, 2)

    return pts


def trim_small_components(mask, min_component_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label] = 255

    return cleaned


def inset_polygon(poly, shape, inset_px):
    if inset_px <= 0:
        return poly

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    kernel_size = 2 * inset_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inset_mask = cv2.erode(mask, kernel)
    contours, _ = cv2.findContours(inset_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return poly

    return max(contours, key=cv2.contourArea)


def outset_polygon(poly, shape, outset_px):
    if outset_px <= 0:
        return poly

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    kernel_size = 2 * outset_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    outset_mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(outset_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return poly

    return max(contours, key=cv2.contourArea)


def build_tolerant_polygon(strict_poly, shape, margin_px, max_points):
    tolerant = outset_polygon(strict_poly, shape, margin_px)
    return simplify_polygon(tolerant, max_points=max_points)


def build_field_polygon(
    frame,
    max_points=6,
    lower_green=(35, 35, 35),
    upper_green=(90, 255, 220),
    min_row_green_frac=0.12,
    left_percentile=3.0,
    right_percentile=97.0,
    min_component_area_frac=0.002,
    min_field_area_frac=0.12,
    max_field_area_frac=0.90,
    inset_px=35,
):
    h, w = frame.shape[:2]
    frame_area = h * w

    green_mask = build_green_mask(frame, lower_green, upper_green)
    green_mask = trim_small_components(
        green_mask,
        min_component_area=max(1, int(frame_area * min_component_area_frac)),
    )

    envelope = build_row_envelope(
        green_mask,
        min_row_green_frac=min_row_green_frac,
        left_percentile=left_percentile,
        right_percentile=right_percentile,
    )

    if envelope is None:
        return None, green_mask, 0, 0.0

    # A rectangular field remains convex under perspective projection. This
    # bridges worn grass instead of creating impossible inward notches.
    field_hull = cv2.convexHull(envelope)
    field_hull = inset_polygon(field_hull, green_mask.shape, inset_px)
    simple = simplify_polygon(field_hull, max_points=max_points)

    area = cv2.contourArea(simple)

    if area < frame_area * min_field_area_frac:
        return None, green_mask, int(area), 0.0

    # Reject masks that are basically the whole frame.
    if area > frame_area * max_field_area_frac:
        return None, green_mask, int(area), 0.0

    filled = np.zeros_like(green_mask)
    cv2.fillPoly(filled, [simple], 255)

    confidence = min(1.0, area / (frame_area * 0.55))

    return simple, filled, int(area), confidence


def build_segmented_field_polygon(frame, model, max_points=6, conf=0.25, imgsz=960, inset_px=0):
    h, w = frame.shape[:2]
    frame_area = h * w
    result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]

    if result.masks is None or not result.masks.xy:
        return None, np.zeros((h, w), dtype=np.uint8), 0, 0.0

    candidates = []
    for index, points in enumerate(result.masks.xy):
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        if len(contour) < 3:
            continue
        box_conf = float(result.boxes.conf[index]) if result.boxes is not None else conf
        candidates.append((cv2.contourArea(contour) * box_conf, box_conf, contour))

    if not candidates:
        return None, np.zeros((h, w), dtype=np.uint8), 0, 0.0

    _, model_confidence, contour = max(candidates, key=lambda item: item[0])
    field_hull = cv2.convexHull(contour)
    field_hull = inset_polygon(field_hull, (h, w), inset_px)
    simple = simplify_polygon(field_hull, max_points=max_points)
    area = cv2.contourArea(simple)

    if area < frame_area * 0.10 or area > frame_area * 0.95:
        return None, np.zeros((h, w), dtype=np.uint8), int(area), 0.0

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(filled, [simple], 255)
    return simple, filled, int(area), model_confidence


def save_debug(frame, strict_poly, tolerant_poly, mask, out_path):
    debug = frame.copy()

    if strict_poly is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [strict_poly], (0, 120, 0))
        debug = cv2.addWeighted(frame, 0.70, overlay, 0.30, 0)
        cv2.polylines(debug, [strict_poly], True, (0, 255, 0), 4)

    if tolerant_poly is not None:
        cv2.polylines(debug, [tolerant_poly], True, (0, 200, 255), 3)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([
        cv2.resize(frame, None, fx=0.5, fy=0.5),
        cv2.resize(mask_bgr, None, fx=0.5, fy=0.5),
        cv2.resize(debug, None, fx=0.5, fy=0.5),
    ])

    cv2.imwrite(str(out_path), combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--sample_every", type=int, default=30)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--max_points", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--left_percentile", type=float, default=3.0)
    parser.add_argument("--right_percentile", type=float, default=97.0)
    parser.add_argument("--min_row_green_frac", type=float, default=0.12)
    parser.add_argument("--min_component_area_frac", type=float, default=0.002)
    parser.add_argument("--min_field_area_frac", type=float, default=0.12)
    parser.add_argument("--max_field_area_frac", type=float, default=0.90)
    parser.add_argument("--inset_px", type=int, default=35)
    parser.add_argument("--green_h_min", type=int, default=35)
    parser.add_argument("--green_h_max", type=int, default=90)
    parser.add_argument("--green_s_min", type=int, default=35)
    parser.add_argument("--green_s_max", type=int, default=255)
    parser.add_argument("--green_v_min", type=int, default=35)
    parser.add_argument("--green_v_max", type=int, default=220)
    parser.add_argument("--segmentation_model", default=None)
    parser.add_argument("--segmentation_conf", type=float, default=0.25)
    parser.add_argument("--segmentation_imgsz", type=int, default=960)
    parser.add_argument("--segmentation_inset_px", type=int, default=0)
    parser.add_argument(
        "--tolerant_margin_px",
        type=int,
        default=15,
        help="Pixels to expand the strict polygon for tolerant player filtering.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segmentation_model = None
    if args.segmentation_model:
        from ultralytics import YOLO

        segmentation_model = YOLO(args.segmentation_model)

    rows = []
    sampled = 0
    success = 0
    frame_idx = 0

    while True:
        if args.max_frames is not None and frame_idx > args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_every == 0:
            if segmentation_model is not None:
                poly, mask, area, confidence = build_segmented_field_polygon(
                    frame,
                    segmentation_model,
                    max_points=args.max_points,
                    conf=args.segmentation_conf,
                    imgsz=args.segmentation_imgsz,
                    inset_px=args.segmentation_inset_px,
                )
            else:
                poly, mask, area, confidence = build_field_polygon(
                    frame,
                    max_points=args.max_points,
                    lower_green=(args.green_h_min, args.green_s_min, args.green_v_min),
                    upper_green=(args.green_h_max, args.green_s_max, args.green_v_max),
                    min_row_green_frac=args.min_row_green_frac,
                    left_percentile=args.left_percentile,
                    right_percentile=args.right_percentile,
                    min_component_area_frac=args.min_component_area_frac,
                    min_field_area_frac=args.min_field_area_frac,
                    max_field_area_frac=args.max_field_area_frac,
                    inset_px=args.inset_px,
                )

            if poly is not None:
                strict_poly = poly
                tolerant_poly = build_tolerant_polygon(
                    strict_poly,
                    mask.shape,
                    margin_px=args.tolerant_margin_px,
                    max_points=args.max_points,
                )
                strict_polygon = polygon_string(strict_poly)
                tolerant_polygon = polygon_string(tolerant_poly)
                num_points = len(poly)
                status = "field_detected"
                success += 1
            else:
                strict_poly = None
                tolerant_poly = None
                strict_polygon = ""
                tolerant_polygon = ""
                num_points = 0
                status = "no_field_detected"

            rows.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3) if fps else "",
                "field_polygon": strict_polygon,
                "strict_field_polygon": strict_polygon,
                "tolerant_field_polygon": tolerant_polygon,
                "num_points": num_points,
                "field_area_px": area,
                "confidence": round(confidence, 3),
                "status": status,
            })

            if debug_dir:
                out_path = debug_dir / f"field_mask_frame_{frame_idx:06d}.jpg"
                save_debug(frame, strict_poly, tolerant_poly, mask, out_path)

            sampled += 1

        frame_idx += 1

    cap.release()

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time_sec",
                "field_polygon",
                "strict_field_polygon",
                "tolerant_field_polygon",
                "num_points",
                "field_area_px",
                "confidence",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAUTO FIELD MASK REPORT")
    print("----------------------")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}" if fps else "FPS: unknown")
    print(f"Total video frames: {total_frames}")
    print(f"Sample every: {args.sample_every} frames")
    print(f"Mask source: {'segmentation model' if segmentation_model else 'color/geometry fallback'}")
    print(f"Sampled frames processed: {sampled}")
    print(f"Successful field masks: {success}")
    print(f"Failed field masks: {sampled - success}")
    print(f"Output CSV: {output_csv}")

    if debug_dir:
        print(f"Debug images: {debug_dir}")


if __name__ == "__main__":
    main()
