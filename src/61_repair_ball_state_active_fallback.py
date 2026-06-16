#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


NON_ACTIVE_DECISIONS = {"high_confidence_non_active", "likely_non_active"}


def clean_id(value):
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def ensure_player_geometry(players):
    players = players.copy()
    players["frame"] = players["frame"].astype(int)
    players["_player_id_clean"] = players["player_id"].map(clean_id)
    if "foot_x" not in players.columns:
        players["foot_x"] = (players["x1"] + players["x2"]) / 2
    if "foot_y" not in players.columns:
        players["foot_y"] = players["y2"]
    players["box_width_px"] = players["x2"] - players["x1"]
    players["box_height_px"] = players["y2"] - players["y1"]
    if "field_zone" not in players.columns:
        players["field_zone"] = "unknown"
    return players


def likely_non_active_candidate(player, ball_speed):
    if player.get("field_zone") != "tolerant":
        return False

    staff_band = (
        player["box_width_px"] <= 26
        and player["box_height_px"] <= 66
        and player["confidence"] <= 0.70
        and ball_speed >= 200
    )
    ref_band = (
        player["box_height_px"] >= 85
        and player["_distance_to_ball_px"] <= 40
        and ball_speed <= 130
    )
    return bool(staff_band or ref_band)


def repair_ball_state_with_active_fallback(
    association_csv,
    player_csv,
    output_csv,
    fallback_distance_px=45,
    source_values=None,
    non_active_decisions=None,
):
    association = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    players = ensure_player_geometry(pd.read_csv(player_csv))
    output_csv = Path(output_csv)

    if source_values is None:
        source_values = {"interpolated"}
    else:
        source_values = set(source_values)

    if non_active_decisions is None:
        non_active_decisions = NON_ACTIVE_DECISIONS
    else:
        non_active_decisions = set(non_active_decisions)

    players_by_frame = {
        int(frame): group.copy() for frame, group in players.groupby("frame")
    }

    repaired_rows = []
    repair_count = 0

    for _, row in association.iterrows():
        out = row.to_dict()
        source = str(row.get("source", ""))
        decision = row.get("active_participant_advisory_decision")
        if (
            source not in source_values
            or decision not in non_active_decisions
            or row.get("ball_state") != "in_transit"
        ):
            out["ball_state_active_fallback_repair"] = False
            out["ball_state_active_fallback_reason"] = ""
            repaired_rows.append(out)
            continue

        frame_players = players_by_frame.get(int(row["frame"]))
        if frame_players is None or frame_players.empty:
            out["ball_state_active_fallback_repair"] = False
            out["ball_state_active_fallback_reason"] = "no_frame_players"
            repaired_rows.append(out)
            continue

        candidates = frame_players.copy()
        candidates["_distance_to_ball_px"] = (
            (candidates["foot_x"] - float(row["center_x"])) ** 2
            + (candidates["foot_y"] - float(row["center_y"])) ** 2
        ) ** 0.5
        candidates = candidates.sort_values("_distance_to_ball_px")
        ball_speed = float(row.get("ball_speed_px_per_second") or 0)

        fallback = None
        for _, candidate in candidates.iterrows():
            if candidate["_distance_to_ball_px"] > fallback_distance_px:
                break
            if clean_id(candidate["player_id"]) == clean_id(row.get("nearest_player_id")):
                continue
            if likely_non_active_candidate(candidate, ball_speed):
                continue
            fallback = candidate
            break

        if fallback is None:
            out["ball_state_active_fallback_repair"] = False
            out["ball_state_active_fallback_reason"] = "no_active_fallback_candidate"
            repaired_rows.append(out)
            continue

        repair_count += 1
        out["pre_repair_ball_state"] = row.get("ball_state")
        out["pre_repair_nearest_player_id"] = row.get("nearest_player_id")
        out["pre_repair_nearest_player_distance_px"] = row.get(
            "nearest_player_distance_px"
        )
        out["pre_repair_nearest_player_field_zone"] = row.get(
            "nearest_player_field_zone"
        )
        out["ball_state"] = "controlled"
        out["association_status"] = "active_fallback_control"
        out["nearest_player_id"] = fallback.get("player_id")
        out["nearest_player_distance_px"] = fallback.get("_distance_to_ball_px")
        out["nearest_player_foot_x"] = fallback.get("foot_x")
        out["nearest_player_foot_y"] = fallback.get("foot_y")
        out["nearest_player_field_zone"] = fallback.get("field_zone")
        out["active_participant_advisory_decision"] = "likely_active_fallback"
        out["active_participant_advisory_reason"] = (
            "interpolated_ball_nearest_non_active_active_fallback"
        )
        out["ball_state_active_fallback_repair"] = True
        out["ball_state_active_fallback_reason"] = (
            "interpolated_in_transit_nearest_non_active_active_candidate_close"
        )
        repaired_rows.append(out)

    out_df = pd.DataFrame(repaired_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("BALL STATE ACTIVE FALLBACK REPAIR COMPLETE")
    print("------------------------------------------")
    print(f"Input association: {association_csv}")
    print(f"Player CSV: {player_csv}")
    print(f"Output: {output_csv}")
    print(f"Repaired rows: {repair_count}")
    print("")
    print("Ball states:")
    print(out_df["ball_state"].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair interpolated in-transit rows where the nearest candidate is "
            "likely non-active but a nearby active-looking player is available."
        )
    )
    parser.add_argument("association_csv")
    parser.add_argument("--player-csv", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--fallback-distance-px", type=float, default=45)
    parser.add_argument(
        "--source-values",
        default="interpolated",
        help="Comma-separated source values eligible for repair.",
    )
    parser.add_argument(
        "--non-active-decisions",
        default="high_confidence_non_active,likely_non_active",
        help="Comma-separated advisory decisions eligible for fallback repair.",
    )
    args = parser.parse_args()

    repair_ball_state_with_active_fallback(
        association_csv=args.association_csv,
        player_csv=args.player_csv,
        output_csv=args.output,
        fallback_distance_px=args.fallback_distance_px,
        source_values=[
            value.strip() for value in args.source_values.split(",") if value.strip()
        ],
        non_active_decisions=[
            value.strip()
            for value in args.non_active_decisions.split(",")
            if value.strip()
        ],
    )


if __name__ == "__main__":
    main()
