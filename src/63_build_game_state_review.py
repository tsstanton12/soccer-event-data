#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


LABELS = [
    "live",
    "dead_ball",
    "restart_setup",
    "restart_kick",
    "goal_stoppage",
    "unclear",
]

RESTART_TYPES = [
    "none",
    "free_kick",
    "corner",
    "goal_kick",
    "throw_in",
    "kickoff",
    "penalty",
    "dropped_ball",
    "unclear",
]

STATE_COLORS = {
    "controlled": (60, 220, 60),
    "pending_control": (0, 215, 255),
    "in_transit": (0, 165, 255),
    "loose_or_unclear": (0, 0, 255),
    "unknown": (180, 180, 180),
}


def clean(value):
    if value is None or pd.isna(value):
        return "-"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value) or "-"


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.52):
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


def row_value(row, key, default=None):
    if row is None:
        return default
    return row._asdict().get(key, default)


def draw_panel(frame, row, title, frame_number):
    raw_state = row_value(row, "ball_state", "unknown")
    chain_state = row_value(row, "possession_state", "unknown")
    panel_color = STATE_COLORS.get(chain_state, STATE_COLORS.get(raw_state, STATE_COLORS["unknown"]))
    ball_color = STATE_COLORS.get(raw_state, STATE_COLORS["unknown"])

    if row is not None and pd.notna(row_value(row, "center_x")) and pd.notna(row_value(row, "center_y")):
        bx = int(round(row_value(row, "center_x")))
        by = int(round(row_value(row, "center_y")))
        cv2.circle(frame, (bx, by), 9, ball_color, -1)

        if pd.notna(row_value(row, "nearest_player_foot_x")) and pd.notna(
            row_value(row, "nearest_player_foot_y")
        ):
            px = int(round(row_value(row, "nearest_player_foot_x")))
            py = int(round(row_value(row, "nearest_player_foot_y")))
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
            cv2.line(frame, (bx, by), (px, py), (255, 255, 255), 1)
            draw_label(frame, f"nearest {clean(row_value(row, 'nearest_player_id'))}", (px + 8, py), scale=0.45)

    lines = [
        title,
        f"Frame {frame_number}",
    ]
    if row is not None:
        lines.extend([
            f"Time {float(row_value(row, 'time_seconds')):.2f}s",
            f"Ball: {raw_state}",
            f"Possession: {chain_state} owner={clean(row_value(row, 'possession_player_id'))}",
            f"Segment: {clean(row_value(row, 'possession_segment_id'))}",
            (
                f"Nearest: {clean(row_value(row, 'nearest_player_id'))} "
                f"dist={clean(row_value(row, 'nearest_player_distance_px'))}"
            ),
            f"Speed: {clean(row_value(row, 'ball_speed_px_per_second'))} px/s",
        ])
    else:
        lines.append("No possession row for this frame")

    panel_height = 24 + len(lines) * 23
    cv2.rectangle(frame, (12, 12), (820, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (820, panel_height), panel_color, 2)
    for index, line in enumerate(lines):
        color = panel_color if line.startswith(("Ball:", "Possession:")) else (255, 255, 255)
        draw_label(frame, line, (24, 40 + index * 23), color=color)
    return frame


def infer_priority(row):
    text = " ".join(
        str(row.get(column, ""))
        for column in ["event_impact_label", "corrected_state_or_owner", "review_notes", "question"]
    ).lower()
    if "goal" in text or "free kick" in text or "game is stopped" in text:
        return 0
    if "clearance" in text or "out of bounds" in text or "restart" in text:
        return 1
    if row.get("event_impact_label") in ["wrong_state", "merge", "shift_start"]:
        return 2
    return 3


def build_review_windows(seed_rows, max_clips):
    rows = seed_rows.copy()
    rows["priority"] = rows.apply(infer_priority, axis=1)
    rows = rows.sort_values(["priority", "center_time", "review_number"], kind="stable")

    windows = []
    seen = set()
    for _, row in rows.iterrows():
        center_time = float(row["center_time"])
        review_id = f"game_state_{row['review_id']}"
        if review_id in seen:
            continue
        seen.add(review_id)
        windows.append({
            "source_review_number": row.get("review_number"),
            "source_review_id": row.get("review_id"),
            "source_kind": row.get("kind"),
            "source_event_impact_label": row.get("event_impact_label"),
            "source_notes": row.get("review_notes"),
            "review_id": review_id,
            "center_time": center_time,
            "start_time": max(0.0, center_time - 3.0),
            "end_time": center_time + 3.0,
            "question": "Is the ball live, dead, or in a restart sequence at this moment?",
        })
        if len(windows) >= max_clips:
            break

    return windows


def build_game_state_review(video_path, frames_csv, seed_manifest_csv, output_dir, max_clips=20):
    video_path = Path(video_path)
    frames_csv = Path(frames_csv)
    seed_manifest_csv = Path(seed_manifest_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = pd.read_csv(frames_csv)
    seed_rows = pd.read_csv(seed_manifest_csv)
    frames_by_frame = {int(row.frame): row for row in frames.itertuples(index=False)}
    windows = build_review_windows(seed_rows, max_clips=max_clips)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manifest_rows = []
    for index, item in enumerate(windows, start=1):
        start_frame = max(0, int(round(item["start_time"] * fps)))
        end_frame = int(round(item["end_time"] * fps))
        stem = f"{index:02d}_{item['review_id']}"
        clip_path = output_dir / f"{stem}.mp4"

        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {clip_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_number in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(draw_panel(frame, frames_by_frame.get(frame_number), item["question"], frame_number))
        writer.release()

        center_frame = int(round(item["center_time"] * fps))
        center_image_path = output_dir / f"{stem}_center.jpg"
        cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(
                str(center_image_path),
                draw_panel(frame, frames_by_frame.get(center_frame), item["question"], center_frame),
            )

        manifest_rows.append({
            "review_number": index,
            "review_id": item["review_id"],
            "source_review_number": item["source_review_number"],
            "source_review_id": item["source_review_id"],
            "source_kind": item["source_kind"],
            "source_event_impact_label": item["source_event_impact_label"],
            "start_time": item["start_time"],
            "end_time": item["end_time"],
            "center_time": item["center_time"],
            "clip_path": str(clip_path.resolve()),
            "center_image_path": str(center_image_path.resolve()),
            "question": item["question"],
            "game_state_label": "",
            "restart_type": "",
            "review_notes": "",
            "source_notes": item["source_notes"],
        })

    cap.release()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "game_state_review_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    html_lines = [
        "<!doctype html><html><body><h1>Game-State Review</h1>",
        "<p>Label whether play is live, stopped, or in a restart sequence. "
        "Use the video for context and the center moment as the label target.</p>",
        f"<p><b>Game-state labels:</b> {', '.join(LABELS)}</p>",
        f"<p><b>Restart types:</b> {', '.join(RESTART_TYPES)}</p>",
    ]
    for row in manifest_rows:
        html_lines.append(f"<h2>{row['review_number']}. {row['source_review_id']}</h2>")
        html_lines.append(
            f"<p><b>Question:</b> {row['question']}<br>"
            f"Time: {row['start_time']:.2f}s to {row['end_time']:.2f}s<br>"
            f"Source event-impact label: {clean(row['source_event_impact_label'])}<br>"
            f"Source notes: {clean(row['source_notes'])}</p>"
        )
        html_lines.append(f"<video src='{Path(row['clip_path']).name}' controls width='960'></video>")
        html_lines.append(f"<p><a href='{Path(row['center_image_path']).name}'>center image</a></p>")
    html_lines.append("</body></html>")
    html_path = output_dir / "game_state_review_order.html"
    html_path.write_text("\n".join(html_lines))

    summary_path = output_dir / "game_state_review_summary.md"
    summary_path.write_text("\n".join([
        "# Game-State Review",
        "",
        f"Video: `{video_path}`",
        f"Frames CSV: `{frames_csv}`",
        f"Seed manifest: `{seed_manifest_csv}`",
        f"Manifest: `{manifest_path}`",
        f"HTML order: `{html_path}`",
        "",
        f"Review clips: {len(manifest)}",
        "",
        f"Game-state labels: `{', '.join(LABELS)}`",
        f"Restart types: `{', '.join(RESTART_TYPES)}`",
        "",
        "Save the completed file as `game_state_review_manifest_completed.csv`.",
        "",
    ]))

    print("GAME-STATE REVIEW COMPLETE")
    print("--------------------------")
    print(f"Review clips: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML: {html_path}")
    print("")
    print(manifest[[
        "review_number",
        "source_review_id",
        "source_event_impact_label",
        "start_time",
        "end_time",
    ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Build short review clips for live/dead-ball and restart context."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--seed-manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-clips", type=int, default=20)
    args = parser.parse_args()

    build_game_state_review(
        video_path=args.video,
        frames_csv=args.frames_csv,
        seed_manifest_csv=args.seed_manifest_csv,
        output_dir=args.output_dir,
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
