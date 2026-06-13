import pandas as pd
import numpy as np
import json
import cv2
from pathlib import Path
import argparse


MIN_CONFIDENCE = 0.25
RESTART_CONFIDENCE = 0.45

MAX_ALLOWED_JUMP = 400
MAX_MISSED_FRAMES = 30
MIN_ACCEPT_SCORE = 150

STATIC_DISTANCE_THRESHOLD = 15
STATIC_FRAME_LIMIT = 12

FOOT_ZONE_TOP_RATIO = 0.62
CLEAT_PENALTY = 300
PLAYER_BODY_PENALTY = 100

CONFIDENCE_WEIGHT = 900
PREDICTION_DISTANCE_WEIGHT = 1.6
LAST_DISTANCE_WEIGHT = 0.7
SIZE_PENALTY_WEIGHT = 4.0


def is_inside_field(x, y, field_polygon):
    return cv2.pointPolygonTest(
        field_polygon,
        (float(x), float(y)),
        False
    ) >= 0


def get_players_near_frame(player_df, frame):
    return player_df[
        (player_df["frame"] >= frame - 5) &
        (player_df["frame"] <= frame + 5)
    ]


def player_penalty(ball_row, players):
    bx = ball_row["center_x"]
    by = ball_row["center_y"]

    penalty = 0

    for _, p in players.iterrows():
        px1, py1, px2, py2 = p["x1"], p["y1"], p["x2"], p["y2"]

        inside_x = px1 <= bx <= px2
        inside_body_y = py1 <= by <= py2

        foot_zone_top = py1 + FOOT_ZONE_TOP_RATIO * (py2 - py1)
        inside_foot_y = foot_zone_top <= by <= py2

        if inside_x and inside_foot_y:
            penalty += CLEAT_PENALTY
        elif inside_x and inside_body_y:
            penalty += PLAYER_BODY_PENALTY

    return penalty


def predict_next(history, current_frame):
    if len(history) < 2:
        return None

    last = history[-1]
    prev = history[-2]

    frame_delta = last["frame"] - prev["frame"]
    if frame_delta <= 0:
        return None

    vx = (last["center_x"] - prev["center_x"]) / frame_delta
    vy = (last["center_y"] - prev["center_y"]) / frame_delta

    future_delta = current_frame - last["frame"]

    return {
        "x": last["center_x"] + vx * future_delta,
        "y": last["center_y"] + vy * future_delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, help="Venue/output name, e.g. siena or patriot_league/lehigh")
    parser.add_argument("--ball-csv", required=True, help="CSV from 08_detect_ball_custom_model.py")
    parser.add_argument("--video", default=None, help="Optional video path, used only for output naming")
    parser.add_argument("--player-csv", default="outputs/detections.csv", help="Player detections CSV")
    parser.add_argument("--field-polygon", default=None, help="Optional field polygon JSON")
    parser.add_argument("--use-field-mask", action="store_true", help="Apply field polygon mask if provided")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--restart-conf", type=float, default=RESTART_CONFIDENCE)
    args = parser.parse_args()

    ball_csv = Path(args.ball_csv)
    if not ball_csv.exists():
        raise FileNotFoundError(f"Ball CSV not found: {ball_csv}")

    venue_safe = args.venue.strip("/").replace("\\", "/")
    output_dir = Path("outputs") / venue_safe
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        video_stem = Path(args.video).stem if args.video else ball_csv.stem.replace("_ball_detections", "")
        output_csv = output_dir / f"{video_stem}_conservative_ball_path.csv"

    ball_df = pd.read_csv(ball_csv)
    print("Raw ball rows:", len(ball_df))

    ball_df = ball_df[ball_df["confidence"] >= args.min_conf].copy()
    print("After confidence filter:", len(ball_df))

    if args.use_field_mask:
        if not args.field_polygon:
            raise ValueError("--use-field-mask requires --field-polygon")

        with open(args.field_polygon, "r") as f:
            polygon_data = json.load(f)

        field_polygon = np.array(polygon_data["points"], dtype=np.int32)

        ball_df["inside_field"] = ball_df.apply(
            lambda row: is_inside_field(row["center_x"], row["center_y"], field_polygon),
            axis=1
        )
        ball_df = ball_df[ball_df["inside_field"]].copy()
        print("After field mask:", len(ball_df))
    else:
        print("Field mask skipped")

    ball_df["box_width"] = ball_df["x2"] - ball_df["x1"]
    ball_df["box_height"] = ball_df["y2"] - ball_df["y1"]
    ball_df["box_size"] = (ball_df["box_width"] + ball_df["box_height"]) / 2

    print("Before size filter:", len(ball_df))
    ball_df = ball_df[ball_df["box_size"] <= 140].copy()
    print("After size filter:", len(ball_df))
    print(
    "Unique frames with candidates:",
    ball_df["frame"].nunique()
)

    player_csv = Path(args.player_csv)
    if player_csv.exists():
        player_df = pd.read_csv(player_csv)
        player_df = player_df[player_df["object_type"] == "person"].copy()
        print("Player CSV loaded:", len(player_df))
    else:
        player_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "object_type"])
        print(f"Player CSV not found, skipping player/cleat penalties: {player_csv}")

    ball_df = ball_df.sort_values(["frame", "confidence"], ascending=[True, False])

    chosen_rows = []
    history = []
    tracker_active = False
    static_counter = 0

    rejected_score_count = 0
    reset_count = 0
    restart_success_count = 0
    static_reject_count = 0
    player_penalty_count = 0

    all_frames = sorted(ball_df["frame"].unique())

    for frame in all_frames:
        group = ball_df[ball_df["frame"] == frame].copy()

        if len(group) == 0:
            continue

        if not tracker_active:
            restart_candidates = group[group["confidence"] >= args.restart_conf].copy()

            if len(restart_candidates) == 0:
                continue

            chosen = restart_candidates.sort_values("confidence", ascending=False).iloc[0]
            restart_success_count += 1
            chosen_rows.append(chosen)
            history.append(chosen)

            tracker_active = True
            static_counter = 0
            continue

        last = history[-1]
        frame_gap = frame - last["frame"]

        if frame_gap > MAX_MISSED_FRAMES:
            reset_count += 1

            tracker_active = False
            history = []
            static_counter = 0
            continue

        pred = predict_next(history, frame)

        group["distance_from_last"] = np.sqrt(
            (group["center_x"] - last["center_x"]) ** 2 +
            (group["center_y"] - last["center_y"]) ** 2
        )

        if pred is not None:
            group["distance_from_prediction"] = np.sqrt(
                (group["center_x"] - pred["x"]) ** 2 +
                (group["center_y"] - pred["y"]) ** 2
            )
        else:
            group["distance_from_prediction"] = group["distance_from_last"]

        players = get_players_near_frame(player_df, frame)

        group["player_penalty"] = [
            player_penalty(row, players) for _, row in group.iterrows()
        ]

        player_penalty_count += len(
            group[group["player_penalty"] > 0]
        )

        group = group[
            (group["distance_from_last"] <= MAX_ALLOWED_JUMP) |
            (group["confidence"] >= 0.70)
        ].copy()

        if len(group) == 0:
            continue

        group["score"] = (
            group["confidence"] * CONFIDENCE_WEIGHT
            - group["distance_from_prediction"] * PREDICTION_DISTANCE_WEIGHT
            - group["distance_from_last"] * LAST_DISTANCE_WEIGHT
            - group["player_penalty"]
            - group["box_size"] * SIZE_PENALTY_WEIGHT
        )

        best = group.sort_values("score", ascending=False).iloc[0]

        if best["score"] < MIN_ACCEPT_SCORE:
            rejected_score_count += 1
            continue

        chosen = best

        static_distance = np.sqrt(
            (chosen["center_x"] - last["center_x"]) ** 2 +
            (chosen["center_y"] - last["center_y"]) ** 2
        )

        if static_distance < STATIC_DISTANCE_THRESHOLD:
            static_counter += 1
        else:
            static_counter = 0

        if static_counter >= STATIC_FRAME_LIMIT:
            static_reject_count += 1
            continue

        chosen_rows.append(chosen)
        history.append(chosen)

        if len(history) > 10:
            history = history[-10:]

    filtered_df = pd.DataFrame(chosen_rows)

    for col in [
        "distance_from_last",
        "distance_from_prediction",
        "player_penalty",
        "score",
        "box_width",
        "box_height",
        "box_size",
        "inside_field",
    ]:
        if col in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=[col])

    filtered_df.to_csv(output_csv, index=False)

    print(f"\nInput candidate detections after filters: {len(ball_df)}")
    print(f"Conservative ball path detections: {len(filtered_df)}")

    print(f"Rejected due to score: {rejected_score_count}")
    print(f"Tracker resets: {reset_count}")
    print(f"Successful restarts: {restart_success_count}")
    print(f"Rejected as static: {static_reject_count}")
    print(f"Detections with player penalties: {player_penalty_count}")

    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()