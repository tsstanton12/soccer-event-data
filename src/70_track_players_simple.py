#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def box_iou(a, b):
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def detection_geometry(row):
    width = float(row["x2"]) - float(row["x1"])
    height = float(row["y2"]) - float(row["y1"])
    center_x = float(row.get("center_x", (float(row["x1"]) + float(row["x2"])) / 2))
    center_y = float(row.get("center_y", (float(row["y1"]) + float(row["y2"])) / 2))
    foot_x = float(row.get("foot_x", center_x))
    foot_y = float(row.get("foot_y", float(row["y2"])))
    area = max(1.0, width * height)
    return {
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "area": area,
    }


def add_geometry_columns(detections):
    detections = detections.copy()
    detections["box_width"] = detections["x2"] - detections["x1"]
    detections["box_height"] = detections["y2"] - detections["y1"]
    if "center_x" not in detections.columns:
        detections["center_x"] = (detections["x1"] + detections["x2"]) / 2
    if "center_y" not in detections.columns:
        detections["center_y"] = (detections["y1"] + detections["y2"]) / 2
    if "foot_x" not in detections.columns:
        detections["foot_x"] = (detections["x1"] + detections["x2"]) / 2
    if "foot_y" not in detections.columns:
        detections["foot_y"] = detections["y2"]
    detections["box_area"] = (detections["box_width"] * detections["box_height"]).clip(lower=1.0)
    return detections


def geometry_from_record(row):
    return {
        "width": float(row["box_width"]),
        "height": float(row["box_height"]),
        "center_x": float(row["center_x"]),
        "center_y": float(row["center_y"]),
        "foot_x": float(row["foot_x"]),
        "foot_y": float(row["foot_y"]),
        "area": float(row["box_area"]),
    }


def candidate_score(track, detection, frame_gap, max_distance_px):
    track_geom = track["geom"]
    detection_geom = geometry_from_record(detection)

    dx = detection_geom["foot_x"] - track_geom["foot_x"]
    dy = detection_geom["foot_y"] - track_geom["foot_y"]
    distance = float((dx * dx + dy * dy) ** 0.5)
    height_scale = max(track_geom["height"], detection_geom["height"], 1.0)
    allowed_distance = max_distance_px + frame_gap * max(12.0, height_scale * 0.45)

    iou = box_iou(track["row"], detection)
    area_ratio = max(track_geom["area"], detection_geom["area"]) / max(
        1.0,
        min(track_geom["area"], detection_geom["area"]),
    )
    area_penalty = min(1.0, abs(np.log(area_ratio)))
    score = (distance / allowed_distance) + (1.0 - iou) * 0.35 + area_penalty * 0.20
    return score, distance, allowed_distance, iou


def track_players(
    detections,
    max_missing_frames=8,
    max_distance_px=90.0,
    min_iou=0.02,
    replace_player_id=False,
):
    detections = add_geometry_columns(detections)
    detections = detections.sort_values(["frame", "confidence"], ascending=[True, False]).copy()
    detections["frame"] = detections["frame"].astype(int)

    tracks = {}
    next_track_id = 1
    output_rows = []
    frame_groups = {
        int(frame): group.to_dict("index")
        for frame, group in detections.groupby("frame", sort=True)
    }

    for processed_frame_count, (frame, frame_detections) in enumerate(frame_groups.items(), start=1):
        active_track_ids = [
            track_id
            for track_id, track in tracks.items()
            if frame - track["last_frame"] <= max_missing_frames + 1
        ]

        candidates = []
        for track_id in active_track_ids:
            track = tracks[track_id]
            frame_gap = max(1, frame - track["last_frame"])
            for detection_index, detection in frame_detections.items():
                score, distance, allowed_distance, iou = candidate_score(
                    track,
                    detection,
                    frame_gap=frame_gap,
                    max_distance_px=max_distance_px,
                )
                if distance <= allowed_distance or iou >= min_iou:
                    candidates.append((
                        score,
                        distance,
                        -iou,
                        track_id,
                        detection_index,
                        allowed_distance,
                        iou,
                    ))

        assigned_tracks = set()
        assigned_detections = set()
        assignments = {}
        for score, distance, _neg_iou, track_id, detection_index, allowed_distance, iou in sorted(candidates):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            assignments[detection_index] = {
                "track_id": track_id,
                "track_match_score": score,
                "track_match_distance_px": distance,
                "track_match_allowed_distance_px": allowed_distance,
                "track_match_iou": iou,
            }

        for detection_index, detection in frame_detections.items():
            row = dict(detection)
            original_player_id = row.get("player_id", "")
            if detection_index in assignments:
                assignment = assignments[detection_index]
                track_id = assignment["track_id"]
                track = tracks[track_id]
                track["age_frames"] += 1
                track["last_frame"] = frame
                track["row"] = row
                track["geom"] = geometry_from_record(row)
                track["missed_frames"] = 0
                row.update(assignment)
                row["track_age_frames"] = track["age_frames"]
                row["track_status"] = "matched"
            else:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "last_frame": frame,
                    "row": row,
                    "geom": geometry_from_record(row),
                    "age_frames": 1,
                    "missed_frames": 0,
                }
                row.update({
                    "track_id": track_id,
                    "track_match_score": np.nan,
                    "track_match_distance_px": np.nan,
                    "track_match_allowed_distance_px": np.nan,
                    "track_match_iou": np.nan,
                    "track_age_frames": 1,
                    "track_status": "new",
                })

            row["detector_player_id"] = original_player_id
            if replace_player_id:
                row["player_id"] = int(row["track_id"])
            output_rows.append(row)

        for track_id, track in tracks.items():
            if track["last_frame"] != frame:
                track["missed_frames"] = frame - track["last_frame"]

        if processed_frame_count % 1000 == 0:
            print(
                f"Processed {processed_frame_count}/{len(frame_groups)} frames; "
                f"tracks so far: {next_track_id - 1}"
            )

    tracked = pd.DataFrame(output_rows)
    if len(tracked):
        tracked["track_id"] = tracked["track_id"].astype(int)
    return tracked


def summarize_tracks(tracked):
    if tracked.empty:
        return {
            "detections": 0,
            "tracks": 0,
            "median_track_detections": 0,
            "long_tracks_2s": 0,
            "new_detection_rate": 0,
        }
    lengths = tracked.groupby("track_id").size()
    fps_estimate = 1.0
    if "time_seconds" in tracked.columns:
        frames = tracked[["frame", "time_seconds"]].drop_duplicates().sort_values("frame")
        if len(frames) > 1:
            duration = frames["time_seconds"].max() - frames["time_seconds"].min()
            frame_span = frames["frame"].max() - frames["frame"].min()
            if duration > 0:
                fps_estimate = frame_span / duration
    return {
        "detections": int(len(tracked)),
        "tracks": int(tracked["track_id"].nunique()),
        "median_track_detections": float(lengths.median()),
        "long_tracks_2s": int((lengths >= max(2, round(fps_estimate * 2))).sum()),
        "new_detection_rate": float((tracked["track_status"] == "new").mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assign simple persistent track IDs to frame-level player detections."
    )
    parser.add_argument("player_detections_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-missing-frames", type=int, default=8)
    parser.add_argument("--max-distance-px", type=float, default=90.0)
    parser.add_argument("--min-iou", type=float, default=0.02)
    parser.add_argument(
        "--replace-player-id",
        action="store_true",
        help="Replace player_id with persistent track_id and keep detector_player_id.",
    )
    args = parser.parse_args()

    player_detections = pd.read_csv(args.player_detections_csv)
    required = ["frame", "player_id", "x1", "y1", "x2", "y2", "confidence"]
    missing = [column for column in required if column not in player_detections.columns]
    if missing:
        raise ValueError(f"Player detections missing required columns: {missing}")

    tracked = track_players(
        player_detections,
        max_missing_frames=args.max_missing_frames,
        max_distance_px=args.max_distance_px,
        min_iou=args.min_iou,
        replace_player_id=args.replace_player_id,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracked.to_csv(output, index=False)
    summary = summarize_tracks(tracked)

    print("PLAYER TRACKING COMPLETE")
    print("------------------------")
    print(f"Input detections: {args.player_detections_csv}")
    print(f"Output tracked detections: {output}")
    print(f"Detections: {summary['detections']}")
    print(f"Tracks: {summary['tracks']}")
    print(f"Median detections per track: {summary['median_track_detections']:.1f}")
    print(f"Tracks lasting about 2s or more: {summary['long_tracks_2s']}")
    print(f"New detection rate: {summary['new_detection_rate']:.3f}")
    print(f"player_id column: {'persistent track_id' if args.replace_player_id else 'original detector id'}")


if __name__ == "__main__":
    main()
