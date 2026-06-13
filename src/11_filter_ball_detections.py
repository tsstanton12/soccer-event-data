import pandas as pd
import numpy as np
import json
import cv2
from pathlib import Path

BALL_CSV = "outputs/custom_ball_detections.csv"
PLAYER_CSV = "outputs/detections.csv"
FIELD_POLYGON_PATH = "outputs/field_polygon.json"
OUTPUT_CSV = "outputs/filtered_ball_detections.csv"

MIN_CONFIDENCE = 0.10
BACKGROUND_Y_CUTOFF = 900

MAX_FRAME_GAP = 25
MAX_ALLOWED_JUMP = 260

# Static false-positive filter
STATIC_PIXEL_RADIUS = 12
STATIC_MIN_HITS = 20

# Player/cleat filter
FOOT_ZONE_TOP_RATIO = 0.62
CLEAT_PENALTY = 250
PLAYER_BODY_PENALTY = 80

# Scoring weights
CONFIDENCE_WEIGHT = 900
PREDICTION_DISTANCE_WEIGHT = 1.4
LAST_DISTANCE_WEIGHT = 0.5
SIZE_PENALTY_WEIGHT = 4.0

with open(FIELD_POLYGON_PATH, "r") as f:
    polygon_data = json.load(f)

FIELD_POLYGON = np.array(polygon_data["points"], dtype=np.int32)

def is_inside_field(x, y):
    return cv2.pointPolygonTest(
        FIELD_POLYGON,
        (float(x), float(y)),
        False
    ) >= 0

def remove_static_false_positives(df):
    """
    Removes detections that appear repeatedly in nearly the same location.
    This catches penalty spots, lights, signs, and other fixed objects.
    """
    df = df.copy()
    df["grid_x"] = (df["center_x"] / STATIC_PIXEL_RADIUS).round().astype(int)
    df["grid_y"] = (df["center_y"] / STATIC_PIXEL_RADIUS).round().astype(int)

    location_counts = (
        df.groupby(["grid_x", "grid_y"])
        .size()
        .reset_index(name="hits")
    )

    static_locations = location_counts[
        location_counts["hits"] >= STATIC_MIN_HITS
    ][["grid_x", "grid_y"]]

    if len(static_locations) == 0:
        return df.drop(columns=["grid_x", "grid_y"])

    df = df.merge(
        static_locations.assign(is_static=True),
        on=["grid_x", "grid_y"],
        how="left"
    )

    df["is_static"] = df["is_static"].fillna(False)

    df = df[df["is_static"] == False].copy()

    return df.drop(columns=["grid_x", "grid_y", "is_static"])

ball_df = pd.read_csv(BALL_CSV)
player_df = pd.read_csv(PLAYER_CSV)

ball_df = ball_df[ball_df["confidence"] >= MIN_CONFIDENCE].copy()
ball_df = ball_df[ball_df["center_y"] < BACKGROUND_Y_CUTOFF].copy()

ball_df["inside_field"] = ball_df.apply(
    lambda row: is_inside_field(row["center_x"], row["center_y"]),
    axis=1
)
ball_df = ball_df[ball_df["inside_field"]].copy()

before_static = len(ball_df)
ball_df = remove_static_false_positives(ball_df)
after_static = len(ball_df)

player_df = player_df[player_df["object_type"] == "person"].copy()

ball_df["box_width"] = ball_df["x2"] - ball_df["x1"]
ball_df["box_height"] = ball_df["y2"] - ball_df["y1"]
ball_df["box_size"] = (ball_df["box_width"] + ball_df["box_height"]) / 2

ball_df = ball_df[ball_df["box_size"] <= 80].copy()

def get_players_near_frame(frame):
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

ball_df = ball_df.sort_values(["frame", "confidence"], ascending=[True, False])

chosen_rows = []
history = []

for frame, group in ball_df.groupby("frame"):
    group = group.copy()

    if len(history) == 0:
        chosen = group.sort_values("confidence", ascending=False).iloc[0]
    else:
        last = history[-1]
        frame_gap = frame - last["frame"]

        if frame_gap > MAX_FRAME_GAP:
            chosen = group.sort_values("confidence", ascending=False).iloc[0]
        else:
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

            players = get_players_near_frame(frame)

            penalties = []
            for _, row in group.iterrows():
                penalties.append(player_penalty(row, players))

            group["player_penalty"] = penalties

            group = group[
                (group["distance_from_last"] <= MAX_ALLOWED_JUMP) |
                (group["confidence"] >= 0.65)
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

            chosen = group.sort_values("score", ascending=False).iloc[0]

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

Path("outputs").mkdir(exist_ok=True)
filtered_df.to_csv(OUTPUT_CSV, index=False)

print(f"Original ball detections after basic filters: {before_static}")
print(f"After static-object filter: {after_static}")
print(f"Filtered ball path detections: {len(filtered_df)}")
print(f"Saved to {OUTPUT_CSV}")