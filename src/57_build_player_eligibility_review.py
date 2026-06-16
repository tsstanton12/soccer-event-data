#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd


ZONE_COLORS = {
    "strict": (60, 220, 60),
    "tolerant": (0, 200, 255),
    "off_field": (0, 0, 255),
    "unknown": (180, 180, 180),
}


def clean_id(value):
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.55):
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def nearest_rows_for_frame(players, frame, ball_x, ball_y, limit):
    frame_players = players[players["frame"] == frame].copy()
    if frame_players.empty:
        return frame_players
    if "foot_x" not in frame_players.columns:
        frame_players["foot_x"] = (frame_players["x1"] + frame_players["x2"]) / 2
    if "foot_y" not in frame_players.columns:
        frame_players["foot_y"] = frame_players["y2"]
    frame_players["distance_to_ball_px"] = (
        (frame_players["foot_x"] - ball_x) ** 2
        + (frame_players["foot_y"] - ball_y) ** 2
    ) ** 0.5
    frame_players["box_height_px"] = frame_players["y2"] - frame_players["y1"]
    frame_players["box_width_px"] = frame_players["x2"] - frame_players["x1"]
    return frame_players.sort_values("distance_to_ball_px").head(limit)


def annotate_review_image(frame, association_row, candidate_rows, title):
    ball_x = int(round(association_row["center_x"]))
    ball_y = int(round(association_row["center_y"]))
    cv2.circle(frame, (ball_x, ball_y), 9, (255, 255, 255), -1)
    cv2.circle(frame, (ball_x, ball_y), 13, (0, 0, 0), 2)
    draw_label(frame, "ball", (ball_x + 12, ball_y - 8), (255, 255, 255), 0.5)

    for rank, (_, player) in enumerate(candidate_rows.iterrows(), start=1):
        zone = player.get("field_zone", "unknown")
        color = ZONE_COLORS.get(zone, ZONE_COLORS["unknown"])
        thickness = 4 if rank == 1 else 2
        x1 = int(round(player["x1"]))
        y1 = int(round(player["y1"]))
        x2 = int(round(player["x2"]))
        y2 = int(round(player["y2"]))
        foot_x = int(round(player["foot_x"]))
        foot_y = int(round(player["foot_y"]))
        player_id = clean_id(player.get("player_id"))
        distance = float(player["distance_to_ball_px"])
        height = float(player["box_height_px"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (foot_x, foot_y), 6, color, -1)
        cv2.line(frame, (ball_x, ball_y), (foot_x, foot_y), color, 1)
        draw_label(
            frame,
            f"#{rank} id={player_id} {zone} {distance:.0f}px h={height:.0f}",
            (x1, max(20, y1 - 8)),
            color,
            0.45,
        )

    panel_lines = [
        title,
        f"Frame {int(association_row['frame'])}  Time {float(association_row['time_seconds']):.2f}s",
        f"Ball state: {association_row.get('ball_state', '-')}",
        f"Association: {association_row.get('association_status', '-')}",
        f"Nearest id: {clean_id(association_row.get('nearest_player_id'))}",
        f"Nearest zone: {association_row.get('nearest_player_field_zone', '-')}",
        f"Speed: {float(association_row.get('ball_speed_px_per_second', 0) or 0):.0f}px/s",
    ]

    panel_height = 26 + len(panel_lines) * 24
    cv2.rectangle(frame, (14, 14), (620, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (14, 14), (620, panel_height), (255, 255, 255), 2)
    for index, line in enumerate(panel_lines):
        draw_label(frame, line, (26, 42 + index * 24), (255, 255, 255))

    return frame


def infer_seed_frames(completed_manifest):
    if completed_manifest is None:
        return set()
    manifest = pd.read_csv(completed_manifest)
    notes = manifest["review_notes"].fillna("").str.lower()
    suspicious = manifest[
        notes.str.contains(
            "sideline|referee|substitute|warming up|defender|defensive|closest player"
        )
    ]
    frames = set()
    for _, row in suspicious.iterrows():
        frames.add(int(row["before_frame"]))
        frames.add(int(row["switch_frame"]))
    return frames


def build_player_eligibility_review(
    video_path,
    association_csv,
    players_csv,
    output_dir,
    completed_manifest=None,
    start_time=0,
    end_time=60,
    nearest_limit=5,
    max_reviews=80,
):
    video_path = Path(video_path)
    association_csv = Path(association_csv)
    players_csv = Path(players_csv)
    output_dir = Path(output_dir)

    association = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    players = pd.read_csv(players_csv)
    if "field_zone" not in players.columns:
        players["field_zone"] = "unknown"

    seed_frames = infer_seed_frames(completed_manifest)

    window = association[
        (association["time_seconds"] >= start_time)
        & (association["time_seconds"] <= end_time)
    ].copy()

    def is_suspicious(row):
        return row.get("nearest_player_field_zone") == "tolerant" or (
            row.get("ball_state") == "controlled"
            and pd.notna(row.get("nearest_player_distance_px"))
            and float(row["nearest_player_distance_px"]) <= 35
            and pd.notna(row.get("ball_speed_px_per_second"))
            and float(row["ball_speed_px_per_second"]) >= 250
        )

    selected = []
    seen_frames = set()

    def add_review_row(row):
        frame = int(row["frame"])
        if frame in seen_frames:
            return False
        selected.append(row)
        seen_frames.add(frame)
        return max_reviews is not None and len(selected) >= max_reviews

    # Reviewed trouble spots are the most valuable labels, so reserve the first
    # slots for those before filling the batch with automatically found cases.
    for _, row in window[window["frame"].isin(seed_frames)].iterrows():
        if add_review_row(row):
            break

    if max_reviews is None or len(selected) < max_reviews:
        for _, row in window.iterrows():
            if not is_suspicious(row):
                continue
            if add_review_row(row):
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    manifest_rows = []
    for index, row in enumerate(selected, start=1):
        frame_number = int(row["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue

        candidates = nearest_rows_for_frame(
            players,
            frame_number,
            float(row["center_x"]),
            float(row["center_y"]),
            nearest_limit,
        )
        if candidates.empty:
            continue

        frame = annotate_review_image(
            frame,
            row,
            candidates,
            "Player Eligibility Review",
        )

        nearest = candidates.iloc[0]
        review_id = f"eligibility_{index:04d}"
        filename = (
            f"{review_id}_f{frame_number}_t{float(row['time_seconds']):07.2f}"
            f"_nearest{clean_id(nearest.get('player_id'))}.jpg"
        )
        image_path = images_dir / filename
        cv2.imwrite(str(image_path), frame)

        manifest_rows.append({
            "review_id": review_id,
            "image_path": str(image_path),
            "frame": frame_number,
            "time_seconds": float(row["time_seconds"]),
            "ball_state": row.get("ball_state"),
            "association_status": row.get("association_status"),
            "ball_speed_px_per_second": row.get("ball_speed_px_per_second"),
            "nearest_player_id": nearest.get("player_id"),
            "nearest_distance_px": nearest.get("distance_to_ball_px"),
            "nearest_field_zone": nearest.get("field_zone"),
            "nearest_confidence": nearest.get("confidence"),
            "nearest_box_height_px": nearest.get("box_height_px"),
            "nearest_box_width_px": nearest.get("box_width_px"),
            "nearest_foot_x": nearest.get("foot_x"),
            "nearest_foot_y": nearest.get("foot_y"),
            "eligibility_label": "",
            "review_notes": "",
        })

    cap.release()

    manifest_path = output_dir / "player_eligibility_review_manifest.csv"
    columns = [
        "review_id",
        "image_path",
        "frame",
        "time_seconds",
        "ball_state",
        "association_status",
        "ball_speed_px_per_second",
        "nearest_player_id",
        "nearest_distance_px",
        "nearest_field_zone",
        "nearest_confidence",
        "nearest_box_height_px",
        "nearest_box_width_px",
        "nearest_foot_x",
        "nearest_foot_y",
        "eligibility_label",
        "review_notes",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("PLAYER ELIGIBILITY REVIEW COMPLETE")
    print("----------------------------------")
    print(f"Reviews exported: {len(manifest_rows)}")
    print(f"Images: {images_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build review images for suspicious player eligibility detections."
    )
    parser.add_argument("video")
    parser.add_argument("association_csv")
    parser.add_argument("players_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--completed-transition-manifest", default=None)
    parser.add_argument("--start-time", type=float, default=0)
    parser.add_argument("--end-time", type=float, default=60)
    parser.add_argument("--nearest-limit", type=int, default=5)
    parser.add_argument("--max-reviews", type=int, default=80)
    args = parser.parse_args()

    build_player_eligibility_review(
        video_path=args.video,
        association_csv=args.association_csv,
        players_csv=args.players_csv,
        output_dir=args.output_dir,
        completed_manifest=args.completed_transition_manifest,
        start_time=args.start_time,
        end_time=args.end_time,
        nearest_limit=args.nearest_limit,
        max_reviews=args.max_reviews,
    )


if __name__ == "__main__":
    main()
