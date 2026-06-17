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
    max_carry_forward_seconds=2.0,
    exclude_active_participant_decisions=None,
    instant_confirm_repaired_control=False,
    start_max_speed_px_per_second=None,
    start_max_distance_px=None,
    track_stability_csv=None,
    exclude_track_stability_flags=None,
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
    carry_forward_frames = 0
    carry_forward_start_time = None
    segment_counter = 0
    output_rows = []
    segments = []
    if exclude_active_participant_decisions is None:
        exclude_active_participant_decisions = set()
    else:
        exclude_active_participant_decisions = set(exclude_active_participant_decisions)
    if exclude_track_stability_flags is None:
        exclude_track_stability_flags = set()
    else:
        exclude_track_stability_flags = set(exclude_track_stability_flags)

    track_stability_by_id = {}
    if track_stability_csv:
        stability = pd.read_csv(track_stability_csv)
        required_stability = ["track_id", "stability_flag"]
        missing_stability = [
            column for column in required_stability if column not in stability.columns
        ]
        if missing_stability:
            raise ValueError(
                f"Track stability CSV missing required columns: {missing_stability}"
            )
        track_stability_by_id = {
            str(int(row["track_id"])): row["stability_flag"]
            for _, row in stability.iterrows()
            if pd.notna(row["track_id"])
        }

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
        current_player = None
        current_segment_id = None
        current_segment_start = None

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
        speed = row.get("ball_speed_px_per_second")
        advisory_decision = row.get("active_participant_advisory_decision")
        advisory_excluded = (
            advisory_decision in exclude_active_participant_decisions
            if pd.notna(advisory_decision)
            else False
        )
        track_stability_flag = track_stability_by_id.get(nearest_player, "")
        track_stability_excluded = (
            track_stability_flag in exclude_track_stability_flags
            if track_stability_flag
            else False
        )
        effective_ball_state = (
            "in_transit"
            if row["ball_state"] == "controlled"
            and (advisory_excluded or track_stability_excluded)
            else row["ball_state"]
        )
        controlled_evidence = (
            effective_ball_state == "controlled"
            and nearest_player is not None
            and pd.notna(distance)
            and float(distance) <= max_control_distance_px
            and not advisory_excluded
            and not track_stability_excluded
        )

        transition_reason = ""
        possession_state = effective_ball_state
        possession_player = current_player

        if controlled_evidence:
            required_confirm_frames = (
                1
                if instant_confirm_repaired_control
                and bool(row.get("ball_state_active_fallback_repair", False))
                else min_confirm_frames
            )
            carry_forward_frames = 0
            carry_forward_start_time = None
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

                if pending_count >= required_confirm_frames:
                    start_blocked = (
                        current_player is None
                        and (
                            (
                                start_max_speed_px_per_second is not None
                                and pd.notna(speed)
                                and float(speed) > start_max_speed_px_per_second
                            )
                            or (
                                start_max_distance_px is not None
                                and pd.notna(distance)
                                and float(distance) > start_max_distance_px
                            )
                        )
                    )
                    if start_blocked:
                        possession_state = "pending_control"
                        possession_player = None
                        transition_reason = "new_possession_start_not_stable"
                    else:
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
            if current_player is not None and effective_ball_state in [
                "in_transit",
                "loose_or_unclear",
            ]:
                carry_forward_frames += 1
                if carry_forward_start_time is None:
                    carry_forward_start_time = time_seconds
                carry_forward_seconds = time_seconds - carry_forward_start_time
                if (
                    max_carry_forward_seconds >= 0
                    and carry_forward_seconds > max_carry_forward_seconds
                ):
                    finish_segment(time_seconds, frame, "carry_forward_timeout")
                    transition_reason = "carry_forward_timeout"
                    possession_state = effective_ball_state
                    possession_player = None
                    carry_forward_frames = 0
                    carry_forward_start_time = None
                else:
                    possession_state = effective_ball_state
                    possession_player = current_player
            elif current_player is None:
                possession_state = effective_ball_state
                possession_player = None
                carry_forward_frames = 0
                carry_forward_start_time = None

        out_row = row.to_dict()
        out_row.update({
            "possession_state": possession_state,
            "possession_player_id": possession_player,
            "possession_segment_id": current_segment_id,
            "pending_player_id": pending_player,
            "pending_confirm_frames": pending_count,
            "carry_forward_frames": carry_forward_frames,
            "carry_forward_seconds": (
                0.0
                if carry_forward_start_time is None
                else time_seconds - carry_forward_start_time
            ),
            "transition_reason": transition_reason,
            "active_participant_advisory_excluded_from_control": advisory_excluded,
            "track_stability_flag": track_stability_flag,
            "track_stability_excluded_from_control": track_stability_excluded,
            "effective_ball_state_for_possession": effective_ball_state,
        })
        output_rows.append(out_row)

    if current_player is not None and len(df):
        last = df.iloc[-1]
        finish_segment(float(last["time_seconds"]), int(last["frame"]), "end_of_input")

    frames_out = pd.DataFrame(output_rows)
    segments_out = pd.DataFrame(
        segments,
        columns=[
            "segment_id",
            "player_id",
            "start_time",
            "end_time",
            "duration_seconds",
            "start_frame",
            "end_frame",
            "end_reason",
        ],
    )

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
    parser.add_argument(
        "--max-carry-forward-seconds",
        type=float,
        default=2.0,
        help=(
            "Seconds to keep the prior controller through in-transit/loose frames "
            "before ending the possession. Use a negative value to disable timeout."
        ),
    )
    parser.add_argument(
        "--exclude-active-participant-decisions",
        default="",
        help=(
            "Comma-separated advisory decisions that should not count as controlled "
            "possession evidence, for example 'high_confidence_non_active'. "
            "Requires association rows from src/60_score_active_participants.py."
        ),
    )
    parser.add_argument(
        "--instant-confirm-repaired-control",
        action="store_true",
        help=(
            "Immediately confirm rows repaired by the active-fallback ball-state "
            "pass as controlled possession. This keeps normal confirmation "
            "thresholds unchanged for all other rows."
        ),
    )
    parser.add_argument(
        "--start-max-speed",
        type=float,
        default=None,
        help=(
            "Optional maximum ball speed for starting a brand-new possession "
            "from no current owner. New-owner switches during an active chain are "
            "not affected."
        ),
    )
    parser.add_argument(
        "--start-max-distance",
        type=float,
        default=None,
        help=(
            "Optional maximum ball-to-player distance for starting a brand-new "
            "possession from no current owner. New-owner switches during an "
            "active chain are not affected."
        ),
    )
    parser.add_argument(
        "--track-stability-csv",
        default=None,
        help=(
            "Optional per-track stability summary from "
            "src/77_evaluate_player_track_stability.py."
        ),
    )
    parser.add_argument(
        "--exclude-track-stability-flags",
        default="",
        help=(
            "Comma-separated stability_flag values that should not count as "
            "controlled possession evidence, for example "
            "'short_track,jumpy_track,gappy_track'."
        ),
    )
    args = parser.parse_args()

    exclude_active_participant_decisions = [
        decision.strip()
        for decision in args.exclude_active_participant_decisions.split(",")
        if decision.strip()
    ]
    exclude_track_stability_flags = [
        flag.strip()
        for flag in args.exclude_track_stability_flags.split(",")
        if flag.strip()
    ]

    build_possession_chains(
        association_csv=args.association_csv,
        output_frames_csv=args.output_frames,
        output_segments_csv=args.output_segments,
        fps=args.fps,
        min_confirm_frames=args.min_confirm_frames,
        max_control_distance_px=args.max_control_distance_px,
        max_carry_forward_seconds=args.max_carry_forward_seconds,
        exclude_active_participant_decisions=exclude_active_participant_decisions,
        instant_confirm_repaired_control=args.instant_confirm_repaired_control,
        start_max_speed_px_per_second=args.start_max_speed,
        start_max_distance_px=args.start_max_distance,
        track_stability_csv=args.track_stability_csv,
        exclude_track_stability_flags=exclude_track_stability_flags,
    )


if __name__ == "__main__":
    main()
