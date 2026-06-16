#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


TEAM_COLORS = {
    "team_1": (255, 80, 80),
    "team_2": (80, 220, 255),
    "unknown": (180, 180, 180),
}


def clean_id(value):
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(frame, text, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def build_review_images(
    review_name,
    video_path,
    player_csv,
    segment_csv,
    output_dir,
    order_start,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(player_csv)
    segments = pd.read_csv(segment_csv).sort_values(["start_time", "end_time"])
    players["frame"] = players["frame"].astype(int)
    players["player_id_clean"] = players["player_id"].map(clean_id)

    rows = []
    order = order_start
    for _, segment in segments.iterrows():
        order += 1
        start_frame = int(segment.get("start_frame", round(float(segment["start_time"]) * 30)))
        end_frame = int(segment.get("end_frame", start_frame))
        frame_number = int(round((start_frame + end_frame) / 2))
        player_id = clean_id(segment["player_id"])
        team = str(segment.get("team", "unknown") or "unknown")
        color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])

        frame = read_frame(video_path, frame_number)
        frame_players = players[players["frame"] == frame_number]
        target = frame_players[frame_players["player_id_clean"] == player_id]
        if target.empty:
            nearest_frame = players.iloc[(players["frame"] - frame_number).abs().argsort()[:1]]["frame"].iloc[0]
            frame_players = players[players["frame"] == int(nearest_frame)]
            target = frame_players[frame_players["player_id_clean"] == player_id]

        for _, player in frame_players.iterrows():
            x1, y1, x2, y2 = [int(round(float(player[c]))) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

        if not target.empty:
            player = target.iloc[0]
            x1, y1, x2, y2 = [int(round(float(player[c]))) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            draw_label(frame, f"id {player_id} {team}", (x1, max(24, y1 - 8)), color=color, scale=0.7)

        panel_lines = [
            f"{review_name} / {segment['segment_id']}",
            f"time {float(segment['start_time']):.2f}-{float(segment['end_time']):.2f}s  frame {frame_number}",
            f"assigned team: {team}",
            f"vote share: {float(segment.get('team_vote_share', 0)):.2f}",
            f"vote frames: {int(segment.get('team_vote_frames', 0))}/{int(segment.get('team_vote_total_frames', 0))}",
        ]
        cv2.rectangle(frame, (10, 10), (610, 165), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (610, 165), color, 2)
        for index, line in enumerate(panel_lines):
            draw_label(frame, line, (24, 40 + index * 24), color=color if index == 2 else (255, 255, 255))

        filename = f"{order:03d}_{review_name}_{segment['segment_id']}_{team}.jpg"
        image_path = image_dir / filename
        cv2.imwrite(str(image_path), frame)

        rows.append({
            "review_order": order,
            "review_name": review_name,
            "segment_id": segment["segment_id"],
            "video": video_path.name,
            "image_path": str(image_path),
            "start_time": float(segment["start_time"]),
            "end_time": float(segment["end_time"]),
            "review_frame": frame_number,
            "player_id": player_id,
            "assigned_team": team,
            "team_vote_share": float(segment.get("team_vote_share", 0)),
            "team_vote_frames": int(segment.get("team_vote_frames", 0)),
            "team_vote_total_frames": int(segment.get("team_vote_total_frames", 0)),
            "review_label": "",
            "corrected_team": "",
            "review_notes": "",
        })

    return rows, order


def main():
    parser = argparse.ArgumentParser(
        description="Build ordered still-frame review images for team classification."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--clip",
        action="append",
        nargs=4,
        metavar=("NAME", "VIDEO", "PLAYERS", "SEGMENTS_WITH_TEAM"),
        required=True,
        help="Clip specification. Repeat for multiple clips.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    order = 0
    for name, video, players, segments in args.clip:
        rows, order = build_review_images(
            review_name=name,
            video_path=video,
            player_csv=players,
            segment_csv=segments,
            output_dir=output_dir,
            order_start=order,
        )
        manifest_rows.extend(rows)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "team_classification_review_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("TEAM CLASSIFICATION REVIEW READY")
    print("--------------------------------")
    print(f"Review items: {len(manifest)}")
    print(f"Images: {output_dir / 'images'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
