#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def build_possession_chains(
    association_csv,
    output_frames_csv,
    output_segments_csv,
    fps=25,
    min_confirm_frames=5,
    max_control_distance_px=80,
):
    association_csv = Path(association_csv)
    output_frames_csv = Path(output_frames_csv)
    output_segments_csv = Path(output_segments_csv)

    df = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    if "time_seconds" not in df.columns:
        df["time_seconds"] = df["frame"] / fps

    required = [
        "frame",
        "time_seconds",
        "nearest_player_id",
        "nearest_player_distance_px",
        "ball_state",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required association columns: {missing}")

    current_player = None
    current_segment_id = None
    current_segment_start = None
    pending_player = None
    pending_count = 0
    segment_counter = 0
    output_rows = []
    segments = []

    def finish_segment(end_time, end_frame, reason):
        nonlocal current_player, current_segment_id, current_segment_start
        if current_segment_id is None:
            return
        segments.append({
            "segment_id": current_segment_id,
            "player_id": current_player,
            "start_time": current_segment_start["time"],
            "end_time": end_time,
            "duration_seconds": max(0.0, end_time - current_segment_start["time"]),
            "start_frame": current_segment_start["frame"],
            "end_frame": end_frame,
            "end_reason": reason,
        })

    def start_segment(player_id, row):
        nonlocal current_player, current_segment_id, current_segment_start, segment_counter
        segment_counter += 1
        current_player = player_id
        current_segment_id = f"possession_{segment_counter:04d}"
        current_segment_start = {
            "time": float(row["time_seconds"]),
            "frame": int(row["frame"]),
        }

    for _, row in df.iterrows():
        frame = int(row["frame"])
        time_seconds = float(row["time_seconds"])
        nearest_player = row.get("nearest_player_id")
        if pd.isna(nearest_player):
            nearest_player = None
        else:
            nearest_player = str(int(nearest_player)) if float(nearest_player).is_integer() else str(nearest_player)

        distance = row.get("nearest_player_distance_px")
        controlled_evidence = (
            row["ball_state"] == "controlled"
            and nearest_player is not None
            and pd.notna(distance)
            and float(distance) <= max_control_distance_px
        )

        transition_reason = ""
        possession_state = row["ball_state"]
        possession_player = current_player

        if controlled_evidence:
            if nearest_player == current_player:
                pending_player = None
                pending_count = 0
                possession_state = "controlled"
                possession_player = current_player
            else:
                if nearest_player == pending_player:
                    pending_count += 1
                else:
                    pending_player = nearest_player
                    pending_count = 1

                if pending_count >= min_confirm_frames:
                    if current_player is not None:
                        finish_segment(time_seconds, frame, "confirmed_new_controller")
                    start_segment(nearest_player, row)
                    transition_reason = "confirmed_new_controller"
                    pending_player = None
                    pending_count = 0
                    possession_state = "controlled"
                    possession_player = current_player
                elif current_player is None:
                    possession_state = "pending_control"
                    possession_player = None
                else:
                    possession_state = "in_transit"
                    possession_player = current_player
                    transition_reason = "candidate_controller_not_confirmed"
        else:
            pending_player = None
            pending_count = 0
            if current_player is not None and row["ball_state"] in [
                "in_transit",
                "loose_or_unclear",
            ]:
                possession_state = row["ball_state"]
                possession_player = current_player
            elif current_player is None:
                possession_state = row["ball_state"]
                possession_player = None

        out_row = row.to_dict()
        out_row.update({
            "possession_state": possession_state,
            "possession_player_id": possession_player,
            "possession_segment_id": current_segment_id,
            "pending_player_id": pending_player,
            "pending_confirm_frames": pending_count,
            "transition_reason": transition_reason,
        })
        output_rows.append(out_row)

    if current_player is not None and len(df):
        last = df.iloc[-1]
        finish_segment(float(last["time_seconds"]), int(last["frame"]), "end_of_input")

    frames_out = pd.DataFrame(output_rows)
    segments_out = pd.DataFrame(segments)

    output_frames_csv.parent.mkdir(parents=True, exist_ok=True)
    output_segments_csv.parent.mkdir(parents=True, exist_ok=True)
    frames_out.to_csv(output_frames_csv, index=False)
    segments_out.to_csv(output_segments_csv, index=False)

    raw_controlled = df[df["ball_state"] == "controlled"]["nearest_player_id"]
    raw_switches = (raw_controlled != raw_controlled.shift()).sum()
    chain_switches = max(0, len(segments_out) - 1)

    print("POSSESSION CHAINS V1 COMPLETE")
    print("-----------------------------")
    print(f"Input association rows: {len(df)}")
    print(f"Raw controlled nearest-player switches: {int(raw_switches)}")
    print(f"Confirmed possession segments: {len(segments_out)}")
    print(f"Confirmed possession switches: {chain_switches}")
    print(f"Frame output: {output_frames_csv}")
    print(f"Segment output: {output_segments_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Build hysteresis-based possession chains from frame-level ball-player association."
    )
    parser.add_argument("association_csv")
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--output-segments", required=True)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--min-confirm-frames", type=int, default=5)
    parser.add_argument("--max-control-distance-px", type=float, default=80)
    args = parser.parse_args()

    build_possession_chains(
        association_csv=args.association_csv,
        output_frames_csv=args.output_frames,
        output_segments_csv=args.output_segments,
        fps=args.fps,
        min_confirm_frames=args.min_confirm_frames,
        max_control_distance_px=args.max_control_distance_px,
    )


if __name__ == "__main__":
    main()

