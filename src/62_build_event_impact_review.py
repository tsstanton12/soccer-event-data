#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd


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


def row_value(row, key, default=None):
    data = row._asdict()
    return data.get(key, default)


def draw_panel(frame, row, title, frame_number):
    if row is None:
        lines = [title, f"Frame {frame_number}", "No association row"]
        cv2.rectangle(frame, (12, 12), (640, 110), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            draw_label(frame, line, (24, 40 + index * 23))
        return frame

    raw_state = row_value(row, "ball_state", "unknown")
    chain_state = row_value(row, "possession_state", "unknown")
    panel_color = STATE_COLORS.get(chain_state, STATE_COLORS.get(raw_state, STATE_COLORS["unknown"]))
    ball_color = STATE_COLORS.get(raw_state, STATE_COLORS["unknown"])

    if pd.notna(row_value(row, "center_x")) and pd.notna(row_value(row, "center_y")):
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
            draw_label(
                frame,
                f"nearest {clean(row_value(row, 'nearest_player_id'))}",
                (px + 8, py),
                (255, 255, 255),
                0.45,
            )

    effective = row_value(row, "effective_ball_state_for_possession", raw_state)
    lines = [
        title,
        f"Frame {int(row_value(row, 'frame'))}  Time {float(row_value(row, 'time_seconds')):.2f}s",
        f"Ball: {raw_state}   Effective: {effective}",
        f"Chain: {chain_state} owner={clean(row_value(row, 'possession_player_id'))}",
        f"Segment: {clean(row_value(row, 'possession_segment_id'))}",
        (
            f"Nearest: {clean(row_value(row, 'nearest_player_id'))}  "
            f"dist={clean(row_value(row, 'nearest_player_distance_px'))}"
        ),
    ]

    reason = row_value(row, "transition_reason")
    if isinstance(reason, str) and reason and reason != "nan":
        lines.append(f"Transition: {reason}")
    if bool(row_value(row, "active_participant_advisory_excluded_from_control", False)):
        lines.append("Advisory: non-active excluded from control")
    if bool(row_value(row, "ball_state_active_fallback_repair", False)):
        lines.append("Repair: active fallback control")

    panel_height = 24 + len(lines) * 23
    cv2.rectangle(frame, (12, 12), (790, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (790, panel_height), panel_color, 2)
    for index, line in enumerate(lines):
        color = panel_color if index in [2, 3] else (255, 255, 255)
        draw_label(frame, line, (24, 40 + index * 23), color, 0.50)

    return frame


def build_review_windows(frames, segments, end_time):
    windows = []
    relevant_segments = segments[segments["start_time"] <= end_time].copy()

    for _, segment in relevant_segments.iterrows():
        segment_id = segment["segment_id"]
        duration = float(segment["duration_seconds"])
        if duration < 0.5:
            windows.append({
                "review_id": f"short_segment_{segment_id}",
                "kind": "short_controlled_segment",
                "segment_id": segment_id,
                "center_time": float(segment["start_time"]),
                "start_time": max(0, float(segment["start_time"]) - 2.0),
                "end_time": float(segment["end_time"]) + 2.0,
                "question": (
                    "Is this very short controlled segment real, or should it "
                    "merge into neighboring possession/in_transit?"
                ),
            })

        windows.append({
            "review_id": f"start_{segment_id}",
            "kind": "segment_start",
            "segment_id": segment_id,
            "center_time": float(segment["start_time"]),
            "start_time": max(0, float(segment["start_time"]) - 1.5),
            "end_time": float(segment["start_time"]) + 1.5,
            "question": "Is this controlled possession start correct?",
        })
        windows.append({
            "review_id": f"end_{segment_id}",
            "kind": "segment_end",
            "segment_id": segment_id,
            "center_time": float(segment["end_time"]),
            "start_time": max(0, float(segment["end_time"]) - 1.5),
            "end_time": float(segment["end_time"]) + 1.5,
            "question": "Is this controlled possession end correct?",
        })

    if "active_participant_advisory_excluded_from_control" in frames.columns:
        excluded = frames[
            (frames["time_seconds"] <= end_time)
            & frames["active_participant_advisory_excluded_from_control"].fillna(False)
        ].copy()
        if len(excluded):
            # Group contiguous excluded runs so sideline/non-player impact is reviewable.
            frames_list = list(excluded["frame"].astype(int))
            run_start = previous = frames_list[0]
            runs = []
            for frame in frames_list[1:]:
                if frame <= previous + 1:
                    previous = frame
                else:
                    runs.append((run_start, previous))
                    run_start = previous = frame
            runs.append((run_start, previous))
            for index, (start_frame, end_frame) in enumerate(runs[:5], start=1):
                run_rows = frames[
                    (frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)
                ]
                start_time = float(run_rows["time_seconds"].min())
                end_time_run = float(run_rows["time_seconds"].max())
                windows.append({
                    "review_id": f"advisory_exclusion_{index:02d}",
                    "kind": "advisory_exclusion",
                    "segment_id": ",".join(
                        sorted(run_rows["possession_segment_id"].dropna().astype(str).unique())
                    ),
                    "center_time": (start_time + end_time_run) / 2,
                    "start_time": max(0, start_time - 1.0),
                    "end_time": end_time_run + 1.0,
                    "question": (
                        "Did excluding the likely non-active nearest candidate improve "
                        "the possession/event decision?"
                    ),
                })

    priority = {
        "short_controlled_segment": 0,
        "segment_start": 1,
        "segment_end": 2,
        "advisory_exclusion": 3,
    }
    unique = []
    seen = set()
    for item in sorted(windows, key=lambda x: (priority.get(x["kind"], 9), x["center_time"])):
        if item["review_id"] in seen:
            continue
        seen.add(item["review_id"])
        unique.append(item)
    return unique


def build_event_impact_review(
    video_path,
    frames_csv,
    segments_csv,
    output_dir,
    end_time=60,
):
    video_path = Path(video_path)
    frames_csv = Path(frames_csv)
    segments_csv = Path(segments_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = pd.read_csv(frames_csv)
    segments = pd.read_csv(segments_csv)
    frames_by_frame = {int(row.frame): row for row in frames.itertuples(index=False)}
    windows = build_review_windows(frames, segments, end_time)

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
        safe_segment = str(item["segment_id"] or "none").replace("/", "-").replace(",", "_")
        stem = f"{index:02d}_{item['kind']}_{safe_segment}"
        clip_path = output_dir / f"{stem}.mp4"
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"avc1"),
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
            annotated = draw_panel(
                frame,
                frames_by_frame.get(frame_number),
                item["question"],
                frame_number,
            )
            writer.write(annotated)
        writer.release()

        center_frame = int(round(item["center_time"] * fps))
        center_image_path = output_dir / f"{stem}_center.jpg"
        cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(
                str(center_image_path),
                draw_panel(
                    frame,
                    frames_by_frame.get(center_frame),
                    item["question"],
                    center_frame,
                ),
            )

        manifest_rows.append({
            "review_number": index,
            "review_id": item["review_id"],
            "kind": item["kind"],
            "segment_id": item["segment_id"],
            "start_time": item["start_time"],
            "end_time": item["end_time"],
            "center_time": item["center_time"],
            "clip_path": str(clip_path.resolve()),
            "center_image_path": str(center_image_path.resolve()),
            "question": item["question"],
            "event_impact_label": "",
            "corrected_state_or_owner": "",
            "review_notes": "",
        })

    cap.release()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "event_impact_review_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    html_lines = [
        "<!doctype html><html><body><h1>Event-Impact Review</h1>",
        "<p>Review only moments likely to change possession segments or downstream event candidates.</p>",
    ]
    for row in manifest_rows:
        html_lines.append(f"<h2>{row['review_number']}. {row['kind']} {row['segment_id']}</h2>")
        html_lines.append(
            f"<p><b>Question:</b> {row['question']}<br>"
            f"Time: {row['start_time']:.2f}s to {row['end_time']:.2f}s</p>"
        )
        html_lines.append(
            f"<video src='{Path(row['clip_path']).name}' controls width='960'></video>"
        )
        html_lines.append(f"<p><a href='{Path(row['center_image_path']).name}'>center image</a></p>")
    html_lines.append("</body></html>")
    html_path = output_dir / "event_impact_review_order.html"
    html_path.write_text("\n".join(html_lines))

    summary_path = output_dir / "event_impact_review_summary.md"
    relevant_segments = segments[segments["start_time"] <= end_time].copy()
    summary_path.write_text("\n".join([
        "# Event-Impact Review",
        "",
        f"Frames CSV: `{frames_csv}`",
        f"Segments CSV: `{segments_csv}`",
        f"Manifest: `{manifest_path}`",
        f"HTML order: `{html_path}`",
        "",
        f"Review clips: {len(manifest)}",
        "",
        "Controlled segments in review window:",
        "",
        relevant_segments.to_string(index=False),
        "",
        "Recommended labels: `correct`, `merge`, `shift_start`, `shift_end`, "
        "`wrong_owner`, `wrong_state`.",
        "",
    ]))

    print("EVENT-IMPACT REVIEW COMPLETE")
    print("----------------------------")
    print(f"Review clips: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML: {html_path}")
    print("")
    print(manifest[["review_number", "review_id", "kind", "segment_id", "start_time", "end_time"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Build short review clips around event-impacting possession moments."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--segments-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--end-time", type=float, default=60)
    args = parser.parse_args()

    build_event_impact_review(
        video_path=args.video,
        frames_csv=args.frames_csv,
        segments_csv=args.segments_csv,
        output_dir=args.output_dir,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
