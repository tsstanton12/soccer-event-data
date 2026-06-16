#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


UNKNOWN_TEAM_VALUES = {"", "unknown", "unk", "none", "nan"}


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


def normalize_team(value):
    text = str(value or "").strip()
    if text.lower() in UNKNOWN_TEAM_VALUES:
        return "unknown"
    return text


def extract_jersey_feature(frame, row):
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(float(row["x1"])))))
    y1 = max(0, min(height - 1, int(round(float(row["y1"])))))
    x2 = max(0, min(width, int(round(float(row["x2"])))))
    y2 = max(0, min(height, int(round(float(row["y2"])))))
    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    torso_x1 = x1 + int(round(box_width * 0.20))
    torso_x2 = x1 + int(round(box_width * 0.80))
    torso_y1 = y1 + int(round(box_height * 0.18))
    torso_y2 = y1 + int(round(box_height * 0.62))
    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Ignore obvious grass pixels when the detector box includes lower body/field.
    green = (hue >= 35) & (hue <= 90) & (saturation >= 35)
    useful = (value >= 35) & ~green
    if useful.sum() < 20:
        useful = value >= 35
    if useful.sum() < 20:
        return None

    pixels = lab[useful].astype(np.float32)
    median_lab = np.median(pixels, axis=0)
    mean_hsv = np.mean(hsv[useful].astype(np.float32), axis=0)
    return {
        "lab_l": float(median_lab[0]),
        "lab_a": float(median_lab[1]),
        "lab_b": float(median_lab[2]),
        "hsv_h": float(mean_hsv[0]),
        "hsv_s": float(mean_hsv[1]),
        "hsv_v": float(mean_hsv[2]),
        "jersey_pixels": int(useful.sum()),
    }


def cluster_features(features, k=2):
    if len(features) < k:
        return np.full(len(features), -1), np.zeros((0, 3), dtype=np.float32)

    matrix = np.array(
        [[f["lab_l"], f["lab_a"], f["lab_b"]] for f in features],
        dtype=np.float32,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )
    _compactness, labels, centers = cv2.kmeans(
        matrix,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.flatten(), centers


def label_clusters(centers):
    if len(centers) == 0:
        return {}
    order = sorted(range(len(centers)), key=lambda idx: centers[idx][0], reverse=True)
    return {cluster: f"team_{rank + 1}" for rank, cluster in enumerate(order)}


def confidence_from_centers(feature, assigned_cluster, centers):
    if assigned_cluster < 0 or len(centers) < 2:
        return 0.0
    vector = np.array([feature["lab_l"], feature["lab_a"], feature["lab_b"]])
    distances = np.linalg.norm(centers - vector, axis=1)
    nearest = float(distances[assigned_cluster])
    sorted_distances = sorted(float(distance) for distance in distances)
    if len(sorted_distances) < 2 or sorted_distances[1] <= 0:
        return 1.0
    margin = max(0.0, sorted_distances[1] - nearest)
    return min(1.0, margin / sorted_distances[1])


def classify_player_rows(video_path, players, min_team_confidence):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows = []
    features = []
    source_indices = []

    for frame_number, group in players.groupby("frame", sort=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = cap.read()
        if not ok:
            continue
        for index, row in group.iterrows():
            feature = extract_jersey_feature(frame, row)
            if feature is None:
                continue
            features.append(feature)
            source_indices.append(index)

    cap.release()

    players = players.copy()
    for column in [
        "jersey_cluster",
        "team_label",
        "team_confidence",
        "jersey_lab_l",
        "jersey_lab_a",
        "jersey_lab_b",
        "jersey_hsv_h",
        "jersey_hsv_s",
        "jersey_hsv_v",
        "jersey_pixels",
    ]:
        players[column] = np.nan
    players["team_label"] = "unknown"

    labels, centers = cluster_features(features, k=2)
    cluster_labels = label_clusters(centers)

    for feature, source_index, cluster in zip(features, source_indices, labels):
        cluster = int(cluster)
        confidence = confidence_from_centers(feature, cluster, centers)
        team_label = cluster_labels.get(cluster, "unknown")
        if confidence < min_team_confidence:
            team_label = "unknown"
        players.loc[source_index, "jersey_cluster"] = cluster
        players.loc[source_index, "team_label"] = team_label
        players.loc[source_index, "team_confidence"] = confidence
        players.loc[source_index, "jersey_lab_l"] = feature["lab_l"]
        players.loc[source_index, "jersey_lab_a"] = feature["lab_a"]
        players.loc[source_index, "jersey_lab_b"] = feature["lab_b"]
        players.loc[source_index, "jersey_hsv_h"] = feature["hsv_h"]
        players.loc[source_index, "jersey_hsv_s"] = feature["hsv_s"]
        players.loc[source_index, "jersey_hsv_v"] = feature["hsv_v"]
        players.loc[source_index, "jersey_pixels"] = feature["jersey_pixels"]

    summary = {
        "feature_count": len(features),
        "cluster_centers_lab": centers.tolist() if len(centers) else [],
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "team_counts": players["team_label"].value_counts(dropna=False).to_dict(),
    }
    return players, summary


def merge_team_to_possession_frames(frames, player_teams):
    frames = frames.copy()
    frames["frame"] = frames["frame"].astype(int)
    frames["possession_player_id_clean"] = frames["possession_player_id"].map(clean_id)

    team_columns = [
        "frame",
        "player_id_clean",
        "team_label",
        "team_confidence",
        "jersey_cluster",
        "jersey_lab_l",
        "jersey_lab_a",
        "jersey_lab_b",
    ]
    player_teams = player_teams.copy()
    player_teams["frame"] = player_teams["frame"].astype(int)
    player_teams["player_id_clean"] = player_teams["player_id"].map(clean_id)

    merged = frames.merge(
        player_teams[team_columns],
        how="left",
        left_on=["frame", "possession_player_id_clean"],
        right_on=["frame", "player_id_clean"],
        suffixes=("", "_player"),
    )
    merged = merged.drop(columns=["player_id_clean"], errors="ignore")
    merged = merged.rename(columns={
        "team_label": "possession_team",
        "team_confidence": "possession_team_confidence",
        "jersey_cluster": "possession_team_cluster",
    })
    merged["possession_team"] = merged["possession_team"].map(normalize_team)
    return merged


def vote_segment_teams(segments, frames_with_team, min_segment_confidence):
    segments = segments.copy()
    frames = frames_with_team.copy()
    frames = frames[
        (frames["possession_segment_id"].notna())
        & (frames["possession_team"].notna())
        & (frames["possession_team"] != "unknown")
    ]

    votes = []
    for segment_id, group in frames.groupby("possession_segment_id"):
        counts = group["possession_team"].value_counts()
        total = int(counts.sum())
        if total == 0:
            continue
        top_team = str(counts.index[0])
        top_count = int(counts.iloc[0])
        share = top_count / total
        mean_confidence = float(
            pd.to_numeric(
                group[group["possession_team"] == top_team]["possession_team_confidence"],
                errors="coerce",
            ).dropna().mean()
        )
        if np.isnan(mean_confidence):
            mean_confidence = 0.0
        votes.append({
            "segment_id": str(segment_id),
            "team": top_team if share >= min_segment_confidence else "unknown",
            "team_vote_share": share,
            "team_vote_frames": top_count,
            "team_vote_total_frames": total,
            "team_mean_detection_confidence": mean_confidence,
        })

    vote_df = pd.DataFrame(votes)
    out = segments.copy()
    if not vote_df.empty:
        out = out.merge(vote_df, on="segment_id", how="left")
    else:
        out["team"] = "unknown"
        out["team_vote_share"] = np.nan
        out["team_vote_frames"] = 0
        out["team_vote_total_frames"] = 0
        out["team_mean_detection_confidence"] = 0.0

    out["team"] = out["team"].map(normalize_team)
    out["team_vote_share"] = pd.to_numeric(out["team_vote_share"], errors="coerce").fillna(0.0)
    out["team_vote_frames"] = pd.to_numeric(out["team_vote_frames"], errors="coerce").fillna(0).astype(int)
    out["team_vote_total_frames"] = pd.to_numeric(out["team_vote_total_frames"], errors="coerce").fillna(0).astype(int)
    out["team_mean_detection_confidence"] = pd.to_numeric(
        out["team_mean_detection_confidence"], errors="coerce"
    ).fillna(0.0)
    return out


def select_relevant_frames(frames, max_frames_per_segment):
    usable = frames[
        (frames["possession_segment_id"].notna())
        & (frames["possession_player_id"].notna())
        & (frames["possession_player_id"].map(clean_id) != "")
    ].copy()
    if "possession_state" in usable.columns:
        usable = usable[usable["possession_state"] == "controlled"].copy()

    selected = set()
    for _segment_id, group in usable.groupby("possession_segment_id"):
        frame_numbers = sorted(group["frame"].astype(int).unique())
        if not frame_numbers:
            continue
        if len(frame_numbers) <= max_frames_per_segment:
            selected.update(frame_numbers)
            continue
        indices = np.linspace(
            0,
            len(frame_numbers) - 1,
            max_frames_per_segment,
            dtype=int,
        )
        selected.update(frame_numbers[index] for index in indices)
    return selected


def write_labels_json(segments, output_json, match_id, clip, half):
    output_json = Path(output_json)
    output_segments = []
    for _, row in segments.sort_values(["start_time", "end_time"]).iterrows():
        if float(row["end_time"]) <= float(row["start_time"]):
            continue
        vote_share = float(row.get("team_vote_share", 0.0))
        mean_confidence = float(row.get("team_mean_detection_confidence", 0.0))
        confidence = min(vote_share, mean_confidence)
        output_segments.append({
            "id": str(row["segment_id"]),
            "start": float(row["start_time"]),
            "end": float(row["end_time"]),
            "state": "controlled",
            "team": normalize_team(row.get("team", "unknown")),
            "player": clean_id(row["player_id"]),
            "confidence": confidence,
            "half": half,
            "source_action_id": "",
            "notes": (
                "Auto-derived possession chain segment with jersey-color team "
                "classification. Player identity is still frame-local."
            ),
            "source_segment_end_reason": str(row.get("end_reason", "")),
            "team_vote_share": vote_share,
            "team_vote_frames": int(row.get("team_vote_frames", 0)),
            "team_vote_total_frames": int(row.get("team_vote_total_frames", 0)),
            "team_mean_detection_confidence": mean_confidence,
            "source_start_frame": int(row["start_frame"]) if pd.notna(row["start_frame"]) else None,
            "source_end_frame": int(row["end_frame"]) if pd.notna(row["end_frame"]) else None,
        })

    payload = {
        "schema_version": "1.0",
        "source_type": "auto_possession_chain_segments_with_jersey_team_classification",
        "review_status": "auto_provisional",
        "match_id": match_id,
        "clip": clip,
        "half": half,
        "team_identity_status": "auto_jersey_color",
        "player_identity_status": "frame_local_not_persistent",
        "segments": output_segments,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify two teams from jersey colors and vote team labels onto "
            "auto possession segments."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--players", required=True, help="Player detection CSV")
    parser.add_argument("--possession-frames", required=True)
    parser.add_argument("--possession-segments", required=True)
    parser.add_argument("--output-players", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--output-segments", required=True)
    parser.add_argument("--output-labels-json", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--min-team-confidence", type=float, default=0.15)
    parser.add_argument("--min-segment-vote-share", type=float, default=0.65)
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=30,
        help="Maximum controlled possession frames sampled per segment.",
    )
    args = parser.parse_args()

    players = pd.read_csv(args.players)
    frames = pd.read_csv(args.possession_frames)
    segments = pd.read_csv(args.possession_segments)

    required_players = ["frame", "player_id", "x1", "y1", "x2", "y2"]
    required_frames = ["frame", "possession_player_id", "possession_segment_id"]
    required_segments = ["segment_id", "player_id", "start_time", "end_time"]
    for name, df, required in [
        ("players", players, required_players),
        ("possession frames", frames, required_frames),
        ("possession segments", segments, required_segments),
    ]:
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    relevant_frames = select_relevant_frames(
        frames,
        max_frames_per_segment=args.max_frames_per_segment,
    )
    if not relevant_frames:
        raise ValueError("No controlled possession frames available for team classification.")
    players = players[players["frame"].astype(int).isin(relevant_frames)].copy()

    player_teams, summary = classify_player_rows(
        args.video,
        players,
        min_team_confidence=args.min_team_confidence,
    )
    frames_with_team = merge_team_to_possession_frames(frames, player_teams)
    segments_with_team = vote_segment_teams(
        segments,
        frames_with_team,
        min_segment_confidence=args.min_segment_vote_share,
    )

    Path(args.output_players).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_frames).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_segments).parent.mkdir(parents=True, exist_ok=True)
    player_teams.to_csv(args.output_players, index=False)
    frames_with_team.to_csv(args.output_frames, index=False)
    segments_with_team.to_csv(args.output_segments, index=False)
    write_labels_json(
        segments_with_team,
        args.output_labels_json,
        match_id=args.match_id,
        clip=args.clip,
        half=args.half,
    )

    print("TEAM CLASSIFICATION COMPLETE")
    print("----------------------------")
    print(f"Video: {args.video}")
    print(f"Player detections classified: {len(player_teams)}")
    print(f"Jersey features used: {summary['feature_count']}")
    print(f"Team counts: {summary['team_counts']}")
    print("")
    print("Segment teams:")
    print(segments_with_team[[
        "segment_id",
        "player_id",
        "team",
        "team_vote_share",
        "team_vote_frames",
        "team_vote_total_frames",
    ]].to_string(index=False))
    print("")
    print(f"Players: {args.output_players}")
    print(f"Frames: {args.output_frames}")
    print(f"Segments: {args.output_segments}")
    print(f"Labels JSON: {args.output_labels_json}")


if __name__ == "__main__":
    main()
