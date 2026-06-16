#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


RULE_PRESETS = {
    "zero_false_positive": {
        "staff_max_width": 26,
        "staff_max_height": 62,
        "staff_max_confidence": 0.45,
        "staff_min_ball_speed": 200,
        "ref_min_height": 90,
        "ref_max_distance": 40,
        "ref_max_ball_speed": 130,
    },
    "balanced_low_fp": {
        "staff_max_width": 26,
        "staff_max_height": 66,
        "staff_max_confidence": 0.70,
        "staff_min_ball_speed": 200,
        "ref_min_height": 85,
        "ref_max_distance": 40,
        "ref_max_ball_speed": 130,
    },
    "higher_recall_review_needed": {
        "staff_max_width": 28,
        "staff_max_height": 66,
        "staff_max_confidence": 0.70,
        "staff_min_ball_speed": 0,
        "ref_min_height": 90,
        "ref_max_distance": 40,
        "ref_max_ball_speed": 130,
    },
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
    players["box_area_px"] = players["box_width_px"] * players["box_height_px"]
    players["box_aspect_ratio"] = players["box_width_px"] / players["box_height_px"]
    if "field_zone" not in players.columns:
        players["field_zone"] = "unknown"
    return players


def frame_dimensions(players, association):
    max_x = max(
        float(players["x2"].max()) if "x2" in players.columns else 0,
        float(association["x2"].max()) if "x2" in association.columns else 0,
        float(association["center_x"].max()) if "center_x" in association.columns else 0,
    )
    max_y = max(
        float(players["y2"].max()) if "y2" in players.columns else 0,
        float(association["y2"].max()) if "y2" in association.columns else 0,
        float(association["center_y"].max()) if "center_y" in association.columns else 0,
    )
    return max_x, max_y


def nearest_candidates_for_frame(players_by_frame, association_row, frame, limit=5):
    frame_players = players_by_frame.get(frame)
    if frame_players is None or frame_players.empty or association_row is None:
        return pd.DataFrame()
    ball_x = float(association_row["center_x"])
    ball_y = float(association_row["center_y"])
    candidates = frame_players.copy()
    candidates["_distance_to_ball_px"] = (
        (candidates["foot_x"] - ball_x) ** 2
        + (candidates["foot_y"] - ball_y) ** 2
    ) ** 0.5
    return candidates.sort_values("_distance_to_ball_px").head(limit)


def compute_local_features(
    players,
    association_by_frame,
    frame,
    foot_x,
    foot_y,
    lookback_frames,
    match_radius_px,
):
    window = players[
        (players["frame"] >= frame - lookback_frames)
        & (players["frame"] <= frame + lookback_frames)
    ].copy()
    if window.empty:
        return {
            "active_participant_local_match_frames": 0,
            "active_participant_local_match_ratio": 0.0,
            "active_participant_local_strict_match_frames": 0,
            "active_participant_local_tolerant_match_frames": 0,
        }

    window["_distance_to_review_foot_px"] = (
        (window["foot_x"] - foot_x) ** 2 + (window["foot_y"] - foot_y) ** 2
    ) ** 0.5
    nearest_by_frame = (
        window.sort_values(["frame", "_distance_to_review_foot_px"])
        .groupby("frame", as_index=False)
        .head(1)
        .copy()
    )
    matched = nearest_by_frame[
        nearest_by_frame["_distance_to_review_foot_px"] <= match_radius_px
    ].copy()

    features = {
        "active_participant_local_match_frames": int(len(matched)),
        "active_participant_local_match_ratio": len(matched)
        / (2 * lookback_frames + 1),
        "active_participant_local_prev_match": int((matched["frame"] < frame).any())
        if len(matched)
        else 0,
        "active_participant_local_next_match": int((matched["frame"] > frame).any())
        if len(matched)
        else 0,
        "active_participant_local_strict_match_frames": int(
            (matched["field_zone"] == "strict").sum()
        )
        if len(matched)
        else 0,
        "active_participant_local_tolerant_match_frames": int(
            (matched["field_zone"] == "tolerant").sum()
        )
        if len(matched)
        else 0,
    }

    if len(matched):
        features["active_participant_local_min_foot_distance_px"] = float(
            matched["_distance_to_review_foot_px"].min()
        )
        features["active_participant_local_median_foot_distance_px"] = float(
            matched["_distance_to_review_foot_px"].median()
        )
    else:
        features["active_participant_local_min_foot_distance_px"] = None
        features["active_participant_local_median_foot_distance_px"] = None

    if len(matched) >= 2:
        ordered = matched.sort_values("frame")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        frame_delta = max(1, int(last["frame"]) - int(first["frame"]))
        displacement = (
            (float(last["foot_x"]) - float(first["foot_x"])) ** 2
            + (float(last["foot_y"]) - float(first["foot_y"])) ** 2
        ) ** 0.5
        features["active_participant_local_displacement_px"] = float(displacement)
        features["active_participant_local_displacement_px_per_frame"] = float(
            displacement / frame_delta
        )
    else:
        features["active_participant_local_displacement_px"] = None
        features["active_participant_local_displacement_px_per_frame"] = None

    ball_distances = []
    for _, match in matched.iterrows():
        assoc = association_by_frame.get(int(match["frame"]))
        if assoc is None:
            continue
        distance = (
            (float(match["foot_x"]) - float(assoc["center_x"])) ** 2
            + (float(match["foot_y"]) - float(assoc["center_y"])) ** 2
        ) ** 0.5
        ball_distances.append(distance)
    if ball_distances:
        features["active_participant_local_min_ball_distance_px"] = float(
            min(ball_distances)
        )
        features["active_participant_local_median_ball_distance_px"] = float(
            pd.Series(ball_distances).median()
        )
    else:
        features["active_participant_local_min_ball_distance_px"] = None
        features["active_participant_local_median_ball_distance_px"] = None

    return features


def score_rule(features, params):
    zone = features.get("active_participant_field_zone") == "tolerant"
    staff = (
        zone
        and features.get("active_participant_box_width_px") <= params["staff_max_width"]
        and features.get("active_participant_box_height_px")
        <= params["staff_max_height"]
        and features.get("active_participant_confidence")
        <= params["staff_max_confidence"]
        and features.get("active_participant_ball_speed_px_per_second")
        >= params["staff_min_ball_speed"]
    )
    ref = (
        zone
        and features.get("active_participant_box_height_px") >= params["ref_min_height"]
        and features.get("active_participant_distance_to_ball_px")
        <= params["ref_max_distance"]
        and features.get("active_participant_ball_speed_px_per_second")
        <= params["ref_max_ball_speed"]
    )
    if staff:
        return True, "staff_band"
    if ref:
        return True, "ref_band"
    return False, ""


def strict_review_signal(features):
    return (
        features.get("active_participant_field_zone") == "strict"
        and features.get("active_participant_box_width_px") <= 26
        and features.get("active_participant_box_height_px") <= 70
        and features.get("active_participant_confidence") <= 0.70
        and features.get("active_participant_distance_to_ball_px") <= 35
        and features.get("active_participant_ball_speed_px_per_second") >= 140
    )


def advisory_decision(features, include_strict_review_signal=False):
    zero_flag, zero_reason = score_rule(features, RULE_PRESETS["zero_false_positive"])
    balanced_flag, balanced_reason = score_rule(
        features, RULE_PRESETS["balanced_low_fp"]
    )
    higher_flag, higher_reason = score_rule(
        features, RULE_PRESETS["higher_recall_review_needed"]
    )
    strict_review = strict_review_signal(features) if include_strict_review_signal else False

    if zero_flag:
        decision = "high_confidence_non_active"
        reason = f"zero_false_positive_{zero_reason}"
        score = 0.95
    elif balanced_flag:
        decision = "likely_non_active"
        reason = f"balanced_low_fp_{balanced_reason}"
        score = 0.80
    elif higher_flag:
        decision = "review_non_active_risk"
        reason = f"higher_recall_{higher_reason}"
        score = 0.60
    elif strict_review:
        decision = "review_non_active_risk"
        reason = "strict_zone_small_close_fast_review"
        score = 0.55
    else:
        decision = "likely_active"
        reason = ""
        score = 0.05

    return {
        "active_participant_zero_fp_flag": zero_flag,
        "active_participant_balanced_low_fp_flag": balanced_flag,
        "active_participant_higher_recall_flag": higher_flag,
        "active_participant_strict_review_flag": strict_review,
        "active_participant_advisory_decision": decision,
        "active_participant_advisory_reason": reason,
        "active_participant_non_active_score": score,
    }


def build_advisory_association(
    player_csv,
    association_csv,
    output_csv,
    output_review_csv=None,
    nearest_limit=5,
    lookback_frames=15,
    match_radius_px=45,
    include_strict_review_signal=False,
):
    players = ensure_player_geometry(pd.read_csv(player_csv))
    association = pd.read_csv(association_csv).sort_values("frame").reset_index(drop=True)
    association["frame"] = association["frame"].astype(int)
    players_by_frame = {
        int(frame): group.copy() for frame, group in players.groupby("frame")
    }
    association_by_frame = {
        int(row["frame"]): row for _, row in association.iterrows()
    }
    frame_width, frame_height = frame_dimensions(players, association)

    output_rows = []
    review_rows = []

    for _, row in association.iterrows():
        frame = int(row["frame"])
        player_id = clean_id(row.get("nearest_player_id"))
        candidates = nearest_candidates_for_frame(
            players_by_frame, row, frame, limit=nearest_limit
        )
        frame_players = players_by_frame.get(frame, pd.DataFrame())

        player = None
        if not frame_players.empty and player_id:
            match = frame_players[frame_players["_player_id_clean"] == player_id]
            if len(match):
                player = match.iloc[0]
        if player is None and len(candidates):
            player = candidates.iloc[0]

        out_row = row.to_dict()
        if player is None:
            out_row.update({
                "active_participant_feature_status": "missing_player_detection",
                "active_participant_advisory_decision": "unknown",
                "active_participant_advisory_reason": "missing_player_detection",
                "active_participant_non_active_score": None,
            })
            output_rows.append(out_row)
            continue

        foot_x = float(player["foot_x"])
        foot_y = float(player["foot_y"])
        candidate_rank = None
        distance_to_ball = None
        if len(candidates):
            ids = list(candidates["_player_id_clean"])
            clean_player_id = clean_id(player["player_id"])
            if clean_player_id in ids:
                candidate_rank = ids.index(clean_player_id) + 1
            candidate_match = candidates[
                candidates["_player_id_clean"] == clean_player_id
            ]
            if len(candidate_match):
                distance_to_ball = float(candidate_match.iloc[0]["_distance_to_ball_px"])

        if distance_to_ball is None:
            distance_to_ball = (
                (foot_x - float(row["center_x"])) ** 2
                + (foot_y - float(row["center_y"])) ** 2
            ) ** 0.5

        features = {
            "active_participant_feature_status": "ok",
            "active_participant_candidate_rank": candidate_rank,
            "active_participant_candidate_count_same_frame": int(len(frame_players)),
            "active_participant_candidate_count_strict_same_frame": int(
                (frame_players["field_zone"] == "strict").sum()
            )
            if not frame_players.empty
            else 0,
            "active_participant_candidate_count_tolerant_same_frame": int(
                (frame_players["field_zone"] == "tolerant").sum()
            )
            if not frame_players.empty
            else 0,
            "active_participant_field_zone": player.get("field_zone"),
            "active_participant_confidence": player.get("confidence"),
            "active_participant_x1": player.get("x1"),
            "active_participant_y1": player.get("y1"),
            "active_participant_x2": player.get("x2"),
            "active_participant_y2": player.get("y2"),
            "active_participant_foot_x": foot_x,
            "active_participant_foot_y": foot_y,
            "active_participant_foot_x_norm": foot_x / frame_width if frame_width else None,
            "active_participant_foot_y_norm": foot_y / frame_height if frame_height else None,
            "active_participant_box_width_px": player.get("box_width_px"),
            "active_participant_box_height_px": player.get("box_height_px"),
            "active_participant_box_area_px": player.get("box_area_px"),
            "active_participant_box_aspect_ratio": player.get("box_aspect_ratio"),
            "active_participant_distance_to_ball_px": distance_to_ball,
            "active_participant_distance_to_ball_norm": distance_to_ball / frame_height
            if frame_height
            else None,
            "active_participant_ball_speed_px_per_second": row.get(
                "ball_speed_px_per_second"
            ),
        }

        features.update(
            compute_local_features(
                players,
                association_by_frame,
                frame,
                foot_x,
                foot_y,
                lookback_frames,
                match_radius_px,
            )
        )
        features.update(
            advisory_decision(
                features,
                include_strict_review_signal=include_strict_review_signal,
            )
        )

        out_row.update(features)
        output_rows.append(out_row)

        if features["active_participant_advisory_decision"] != "likely_active":
            review_rows.append(out_row)

    out = pd.DataFrame(output_rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    review_out = pd.DataFrame(review_rows)
    if output_review_csv:
        output_review_csv = Path(output_review_csv)
        output_review_csv.parent.mkdir(parents=True, exist_ok=True)
        review_out.to_csv(output_review_csv, index=False)

    return out, review_out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add advisory active-participant scores to ball/player association rows."
        )
    )
    parser.add_argument("--player-csv", required=True)
    parser.add_argument("--association-csv", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--review-output",
        default=None,
        help="Optional CSV containing only rows flagged for likely/review non-active risk.",
    )
    parser.add_argument("--nearest-limit", type=int, default=5)
    parser.add_argument("--lookback-frames", type=int, default=15)
    parser.add_argument("--match-radius-px", type=float, default=45)
    parser.add_argument(
        "--include-strict-review-signal",
        action="store_true",
        help=(
            "Also flag strict-zone small/close/fast candidates for review. "
            "This catches known Le Moyne referee cases but is intentionally off "
            "by default because it is noisy on other venues."
        ),
    )
    args = parser.parse_args()

    out, review = build_advisory_association(
        player_csv=args.player_csv,
        association_csv=args.association_csv,
        output_csv=args.output,
        output_review_csv=args.review_output,
        nearest_limit=args.nearest_limit,
        lookback_frames=args.lookback_frames,
        match_radius_px=args.match_radius_px,
        include_strict_review_signal=args.include_strict_review_signal,
    )

    print("ACTIVE PARTICIPANT ADVISORY COMPLETE")
    print("------------------------------------")
    print(f"Association rows: {len(out)}")
    print(f"Output: {args.output}")
    if args.review_output:
        print(f"Review/risk rows: {len(review)}")
        print(f"Review output: {args.review_output}")
    print("")
    print("Advisory decisions:")
    print(out["active_participant_advisory_decision"].value_counts(dropna=False).to_string())
    print("")
    print("Advisory reasons:")
    print(out["active_participant_advisory_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
