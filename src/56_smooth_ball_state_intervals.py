#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def smooth_ball_states(
    association_csv,
    output_csv,
    enter_speed_px_per_second=150,
    enter_distance_px=65,
    core_speed_px_per_second=700,
    core_distance_px=40,
    continuation_speed_px_per_second=75,
    continuation_distance_px=65,
    tight_control_distance_px=22,
    seed_gap_rows=2,
    receiver_confirm_rows=3,
    receiver_strict_zones=None,
    tolerant_receiver_confirm_rows=999,
):
    association_csv = Path(association_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    raw_state = df["ball_state"].astype(str)
    speed = df["ball_speed_px_per_second"].fillna(0)
    distance = df["nearest_player_distance_px"].fillna(9999)

    seed = (
        ((speed >= core_speed_px_per_second) & (distance >= core_distance_px))
        | ((speed >= enter_speed_px_per_second) & (distance >= enter_distance_px))
    )
    seed = seed & ~(distance <= tight_control_distance_px)

    continuation = (
        (speed >= continuation_speed_px_per_second)
        | (distance >= continuation_distance_px)
        | raw_state.eq("in_transit")
    )

    transit_mask = pd.Series(False, index=df.index)
    seed_indices = list(df.index[seed])

    if seed_indices:
        clusters = []
        start = previous = seed_indices[0]
        for index in seed_indices[1:]:
            if index - previous <= seed_gap_rows + 1:
                previous = index
            else:
                clusters.append((start, previous))
                start = previous = index
        clusters.append((start, previous))

        for start, end in clusters:
            left = start
            while left > 0 and bool(continuation.iloc[left - 1]):
                if (
                    distance.iloc[left - 1] <= tight_control_distance_px
                    and raw_state.iloc[left - 1] == "controlled"
                    and start - (left - 1) > 1
                ):
                    break
                left -= 1

            right = end
            while right < len(df) - 1 and bool(continuation.iloc[right + 1]):
                if (
                    distance.iloc[right + 1] <= tight_control_distance_px
                    and raw_state.iloc[right + 1] == "controlled"
                    and (right + 1) - end > 1
                ):
                    break
                right += 1

            transit_mask.iloc[left : right + 1] = True

    smoothed = raw_state.copy()
    smoothed[transit_mask] = "in_transit"
    smoothed[(~transit_mask) & (distance <= 80) & (speed <= 500)] = "controlled"
    smoothed[(~transit_mask) & (distance > 80)] = "loose_or_unclear"

    out = df.copy()
    out["raw_ball_state"] = raw_state
    out["raw_association_status"] = out["association_status"]
    out["ball_state"] = smoothed
    out["ball_state_smoothing_reason"] = "raw"
    out.loc[transit_mask, "ball_state_smoothing_reason"] = "pass_interval"
    out.loc[
        (~transit_mask) & (distance <= 80) & (speed <= 500),
        "ball_state_smoothing_reason",
    ] = "control_distance"
    out.loc[
        (~transit_mask) & (distance > 80),
        "ball_state_smoothing_reason",
    ] = "outside_control_distance"

    if receiver_strict_zones is None:
        receiver_strict_zones = {"strict", "tolerant"}
    else:
        receiver_strict_zones = set(receiver_strict_zones)

    if receiver_confirm_rows > 1:
        states = list(out["ball_state"])
        runs = []
        run_start = 0
        for index in range(1, len(states) + 1):
            if index == len(states) or states[index] != states[run_start]:
                runs.append((run_start, index - 1, states[run_start]))
                run_start = index

        for run_index, (start, end, state) in enumerate(runs):
            if state != "controlled":
                continue
            previous_state = runs[run_index - 1][2] if run_index > 0 else None
            run_length = end - start + 1
            receiver_zone = out.loc[start:end, "nearest_player_field_zone"].mode()
            receiver_zone = receiver_zone.iloc[0] if len(receiver_zone) else ""
            required_rows = (
                receiver_confirm_rows
                if receiver_zone in receiver_strict_zones
                else tolerant_receiver_confirm_rows
            )
            if previous_state == "in_transit" and run_length < required_rows:
                out.loc[start:end, "ball_state"] = "in_transit"
                out.loc[
                    start:end, "ball_state_smoothing_reason"
                ] = "receiver_not_confirmed"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    changed = (out["raw_ball_state"] != out["ball_state"]).sum()
    print("BALL STATE INTERVAL SMOOTHING COMPLETE")
    print("--------------------------------------")
    print(f"Input: {association_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows: {len(out)}")
    print(f"Changed ball-state rows: {changed}")
    print("")
    print("Raw states:")
    print(out["raw_ball_state"].value_counts().to_string())
    print("")
    print("Smoothed states:")
    print(out["ball_state"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Apply interval-level smoothing to raw ball-state classifications."
    )
    parser.add_argument("association_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--enter-speed", type=float, default=150)
    parser.add_argument("--enter-distance", type=float, default=65)
    parser.add_argument("--core-speed", type=float, default=700)
    parser.add_argument("--core-distance", type=float, default=40)
    parser.add_argument("--continuation-speed", type=float, default=75)
    parser.add_argument("--continuation-distance", type=float, default=65)
    parser.add_argument("--tight-control-distance", type=float, default=22)
    parser.add_argument("--seed-gap-rows", type=int, default=2)
    parser.add_argument(
        "--receiver-confirm-rows",
        type=int,
        default=3,
        help=(
            "Minimum controlled rows required after an in-transit interval before "
            "accepting the pass as received. Shorter controlled islands remain in_transit."
        ),
    )
    parser.add_argument(
        "--receiver-strict-zones",
        default="strict,tolerant",
        help=(
            "Comma-separated field zones that can confirm a receiver with "
            "--receiver-confirm-rows. Other zones require --tolerant-receiver-confirm-rows."
        ),
    )
    parser.add_argument(
        "--tolerant-receiver-confirm-rows",
        type=int,
        default=999,
        help=(
            "Rows required for a controlled run in zones outside --receiver-strict-zones "
            "to end an in-transit interval."
        ),
    )
    args = parser.parse_args()

    smooth_ball_states(
        association_csv=args.association_csv,
        output_csv=args.output,
        enter_speed_px_per_second=args.enter_speed,
        enter_distance_px=args.enter_distance,
        core_speed_px_per_second=args.core_speed,
        core_distance_px=args.core_distance,
        continuation_speed_px_per_second=args.continuation_speed,
        continuation_distance_px=args.continuation_distance,
        tight_control_distance_px=args.tight_control_distance,
        seed_gap_rows=args.seed_gap_rows,
        receiver_confirm_rows=args.receiver_confirm_rows,
        receiver_strict_zones=[
            zone.strip()
            for zone in args.receiver_strict_zones.split(",")
            if zone.strip()
        ],
        tolerant_receiver_confirm_rows=args.tolerant_receiver_confirm_rows,
    )


if __name__ == "__main__":
    main()
