#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


NON_ACTIVE_LABELS = {"referee", "substitute_or_staff", "off_field_player"}

DEFAULT_SOURCES = {
    "army": {
        "players": "outputs/field_segmentation_strict_tolerant_evaluation/army_players_on_field.csv",
        "association": "outputs/field_segmentation_strict_tolerant_evaluation/army_ball_player_association.csv",
    },
    "holy_cross": {
        "players": "outputs/active_participant_review/holy_cross_players_on_field_first180s.csv",
        "association": "outputs/active_participant_review/holy_cross_ball_player_association_first180s.csv",
    },
    "lemoyne": {
        "players": "outputs/possession_chain_tuning/lemoyne_player_detections_on_field_hybrid.csv",
        "association": "outputs/possession_chain_tuning/lemoyne_ball_player_association_hybrid_default_control.csv",
    },
    "siena": {
        "players": "outputs/active_participant_review/siena_players_on_field_first120s.csv",
        "association": "outputs/active_participant_review/siena_ball_player_association_first120s.csv",
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


def parse_source(source):
    parts = source.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--source must use 'venue:players_csv:association_csv' format."
        )
    venue, players, association = parts
    return venue, {"players": players, "association": association}


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


def nearest_candidates_for_frame(players, association_row, frame, limit=5):
    frame_players = players[players["frame"] == frame].copy()
    if frame_players.empty or association_row is None:
        return frame_players
    ball_x = float(association_row["center_x"])
    ball_y = float(association_row["center_y"])
    frame_players["_distance_to_ball_px"] = (
        (frame_players["foot_x"] - ball_x) ** 2
        + (frame_players["foot_y"] - ball_y) ** 2
    ) ** 0.5
    return frame_players.sort_values("_distance_to_ball_px").head(limit)


def row_or_none(df):
    if df is None or len(df) == 0:
        return None
    return df.iloc[0]


def compute_local_motion_features(
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
        return {}

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
        "local_match_frames": int(len(matched)),
        "local_match_ratio": len(matched) / (2 * lookback_frames + 1),
        "local_prev_match": int((matched["frame"] < frame).any()),
        "local_next_match": int((matched["frame"] > frame).any()),
        "local_min_foot_distance_px": float(
            matched["_distance_to_review_foot_px"].min()
        )
        if len(matched)
        else None,
        "local_median_foot_distance_px": float(
            matched["_distance_to_review_foot_px"].median()
        )
        if len(matched)
        else None,
        "local_strict_match_frames": int((matched["field_zone"] == "strict").sum())
        if len(matched)
        else 0,
        "local_tolerant_match_frames": int((matched["field_zone"] == "tolerant").sum())
        if len(matched)
        else 0,
    }

    if len(matched) >= 2:
        first = matched.sort_values("frame").iloc[0]
        last = matched.sort_values("frame").iloc[-1]
        frame_delta = max(1, int(last["frame"]) - int(first["frame"]))
        displacement = (
            (float(last["foot_x"]) - float(first["foot_x"])) ** 2
            + (float(last["foot_y"]) - float(first["foot_y"])) ** 2
        ) ** 0.5
        features["local_displacement_px"] = float(displacement)
        features["local_displacement_px_per_frame"] = float(displacement / frame_delta)
    else:
        features["local_displacement_px"] = None
        features["local_displacement_px_per_frame"] = None

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
        features["local_min_ball_distance_px"] = float(min(ball_distances))
        features["local_median_ball_distance_px"] = float(
            pd.Series(ball_distances).median()
        )
    else:
        features["local_min_ball_distance_px"] = None
        features["local_median_ball_distance_px"] = None

    return features


def build_features_for_review(
    review,
    source_map,
    nearest_limit=5,
    lookback_frames=15,
    match_radius_px=45,
):
    source_cache = {}
    feature_rows = []

    for _, review_row in review.iterrows():
        venue = str(review_row["venue"])
        if venue not in source_map:
            raise ValueError(f"No source configured for venue: {venue}")

        if venue not in source_cache:
            source = source_map[venue]
            players = ensure_player_geometry(pd.read_csv(source["players"]))
            association = pd.read_csv(source["association"]).copy()
            association["frame"] = association["frame"].astype(int)
            association_by_frame = {
                int(row["frame"]): row for _, row in association.iterrows()
            }
            width, height = frame_dimensions(players, association)
            source_cache[venue] = {
                "players": players,
                "association": association,
                "association_by_frame": association_by_frame,
                "frame_width_px": width,
                "frame_height_px": height,
            }

        source = source_cache[venue]
        players = source["players"]
        association_by_frame = source["association_by_frame"]
        frame = int(review_row["frame"])
        player_id = clean_id(review_row["nearest_player_id"])
        association_row = association_by_frame.get(frame)

        player_match = players[
            (players["frame"] == frame) & (players["_player_id_clean"] == player_id)
        ]
        player = row_or_none(player_match)

        # Fallback for old manifests where the reviewed candidate was selected
        # from nearest rows but the exact ID format changed.
        candidates = nearest_candidates_for_frame(
            players,
            association_row,
            frame,
            limit=nearest_limit,
        )
        if player is None and len(candidates):
            player = candidates.iloc[0]

        if player is None:
            feature_rows.append({
                **review_row.to_dict(),
                "feature_status": "missing_player_detection",
            })
            continue

        foot_x = float(player["foot_x"])
        foot_y = float(player["foot_y"])
        frame_width = source["frame_width_px"]
        frame_height = source["frame_height_px"]
        rank = None
        distance_to_ball = None
        if len(candidates):
            candidate_ids = list(candidates["_player_id_clean"])
            if player_id in candidate_ids:
                rank = candidate_ids.index(player_id) + 1
            if "_distance_to_ball_px" in candidates.columns:
                match = candidates[candidates["_player_id_clean"] == player_id]
                if len(match):
                    distance_to_ball = float(match.iloc[0]["_distance_to_ball_px"])

        if association_row is not None and distance_to_ball is None:
            distance_to_ball = (
                (foot_x - float(association_row["center_x"])) ** 2
                + (foot_y - float(association_row["center_y"])) ** 2
            ) ** 0.5

        same_frame_players = players[players["frame"] == frame]
        same_zone_count = int((same_frame_players["field_zone"] == player["field_zone"]).sum())

        base = {
            **review_row.to_dict(),
            "feature_status": "ok",
            "truth_binary": "non_active"
            if str(review_row.get("eligibility_label", "")).strip() in NON_ACTIVE_LABELS
            else "active_player",
            "candidate_rank": rank,
            "candidate_count_same_frame": int(len(same_frame_players)),
            "candidate_count_strict_same_frame": int(
                (same_frame_players["field_zone"] == "strict").sum()
            ),
            "candidate_count_tolerant_same_frame": int(
                (same_frame_players["field_zone"] == "tolerant").sum()
            ),
            "candidate_count_same_zone": same_zone_count,
            "candidate_field_zone": player.get("field_zone"),
            "candidate_confidence": player.get("confidence"),
            "candidate_x1": player.get("x1"),
            "candidate_y1": player.get("y1"),
            "candidate_x2": player.get("x2"),
            "candidate_y2": player.get("y2"),
            "candidate_foot_x": foot_x,
            "candidate_foot_y": foot_y,
            "candidate_foot_x_norm": foot_x / frame_width if frame_width else None,
            "candidate_foot_y_norm": foot_y / frame_height if frame_height else None,
            "candidate_box_width_px": player.get("box_width_px"),
            "candidate_box_height_px": player.get("box_height_px"),
            "candidate_box_area_px": player.get("box_area_px"),
            "candidate_box_aspect_ratio": player.get("box_aspect_ratio"),
            "candidate_distance_to_ball_px": distance_to_ball,
            "candidate_distance_to_ball_norm": distance_to_ball / frame_height
            if distance_to_ball is not None and frame_height
            else None,
            "candidate_near_left_edge_norm": foot_x / frame_width if frame_width else None,
            "candidate_near_right_edge_norm": (frame_width - foot_x) / frame_width
            if frame_width
            else None,
            "candidate_near_top_edge_norm": foot_y / frame_height if frame_height else None,
            "candidate_near_bottom_edge_norm": (frame_height - foot_y) / frame_height
            if frame_height
            else None,
        }

        if association_row is not None:
            for col in [
                "ball_state",
                "association_status",
                "ball_speed_px_per_second",
                "nearest_player_field_zone",
                "nearest_player_distance_px",
            ]:
                if col in association_row:
                    base[f"association_{col}"] = association_row[col]

        base.update(
            compute_local_motion_features(
                players,
                association_by_frame,
                frame,
                foot_x,
                foot_y,
                lookback_frames,
                match_radius_px,
            )
        )

        feature_rows.append(base)

    return pd.DataFrame(feature_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build reviewed active-participant feature table."
    )
    parser.add_argument("review_csv")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source mapping as venue:players_csv:association_csv. Can be repeated.",
    )
    parser.add_argument("--nearest-limit", type=int, default=5)
    parser.add_argument("--lookback-frames", type=int, default=15)
    parser.add_argument("--match-radius-px", type=float, default=45)
    args = parser.parse_args()

    source_map = dict(DEFAULT_SOURCES)
    for source in args.source:
        venue, mapping = parse_source(source)
        source_map[venue] = mapping

    review = pd.read_csv(args.review_csv)
    if "venue" not in review.columns:
        raise ValueError("Review CSV must include a venue column.")

    features = build_features_for_review(
        review,
        source_map=source_map,
        nearest_limit=args.nearest_limit,
        lookback_frames=args.lookback_frames,
        match_radius_px=args.match_radius_px,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)

    print("ACTIVE PARTICIPANT FEATURES COMPLETE")
    print("------------------------------------")
    print(f"Review rows: {len(review)}")
    print(f"Feature rows: {len(features)}")
    print(f"Output: {output}")
    print("")
    print("Feature status:")
    print(features["feature_status"].value_counts(dropna=False).to_string())
    if "eligibility_label" in features.columns:
        print("")
        print("Labels:")
        print(features["eligibility_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
