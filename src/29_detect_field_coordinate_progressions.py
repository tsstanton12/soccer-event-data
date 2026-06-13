import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = "outputs/conservative_ball_path_field_coords.csv"
OUTPUT_CSV = "outputs/field_coordinate_progressions.csv"

MAX_GAP_SECONDS_WITHIN_SEQUENCE = 1.5
MIN_SEQUENCE_DURATION = 1.0
MAX_SEQUENCE_DURATION = 12.0

MIN_FIELD_DISTANCE_YARDS = 15
MIN_AVG_SPEED_YARDS_PER_SECOND = 4

ball = pd.read_csv(INPUT_CSV)
ball = ball.sort_values("time_seconds").copy()

# Keep only calibrated points inside or close to field bounds
ball = ball[
    (ball["field_x_yards"] >= -5) &
    (ball["field_x_yards"] <= 125) &
    (ball["field_y_yards"] >= -5) &
    (ball["field_y_yards"] <= 80)
].copy()

ball["time_gap"] = ball["time_seconds"].diff().fillna(0)
ball["new_sequence"] = ball["time_gap"] > MAX_GAP_SECONDS_WITHIN_SEQUENCE
ball["sequence_id"] = ball["new_sequence"].cumsum() + 1

progressions = []

for sequence_id, group in ball.groupby("sequence_id"):
    group = group.sort_values("time_seconds")

    start = group.iloc[0]
    end = group.iloc[-1]

    duration = end["time_seconds"] - start["time_seconds"]

    if duration < MIN_SEQUENCE_DURATION:
        continue

    if duration > MAX_SEQUENCE_DURATION:
        continue

    start_x = start["field_x_yards"]
    start_y = start["field_y_yards"]
    end_x = end["field_x_yards"]
    end_y = end["field_y_yards"]

    dx = end_x - start_x
    dy = end_y - start_y

    field_distance = np.sqrt(dx**2 + dy**2)
    avg_speed = field_distance / duration if duration > 0 else 0

    if field_distance < MIN_FIELD_DISTANCE_YARDS:
        continue

    if avg_speed < MIN_AVG_SPEED_YARDS_PER_SECOND:
        continue

    if abs(dx) > abs(dy) * 1.5:
        movement_shape = "mostly_lengthwise"
    elif abs(dy) > abs(dx) * 1.5:
        movement_shape = "mostly_widthwise"
    else:
        movement_shape = "diagonal"

    progressions.append({
        "sequence_id": int(sequence_id),
        "start_frame": int(start["frame"]),
        "end_frame": int(end["frame"]),
        "start_time": start["time_seconds"],
        "end_time": end["time_seconds"],
        "duration_seconds": duration,
        "num_detections": len(group),
        "start_field_x": start_x,
        "start_field_y": start_y,
        "end_field_x": end_x,
        "end_field_y": end_y,
        "field_dx_yards": dx,
        "field_dy_yards": dy,
        "field_distance_yards": field_distance,
        "avg_speed_yards_per_second": avg_speed,
        "movement_shape": movement_shape,
        "avg_confidence": group["confidence"].mean(),
    })

progression_df = pd.DataFrame(progressions)

Path("outputs").mkdir(exist_ok=True)
progression_df.to_csv(OUTPUT_CSV, index=False)

print("\nFIELD COORDINATE PROGRESSION REPORT")
print("-----------------------------------")
print(f"Input calibrated ball detections after bounds filter: {len(ball)}")
print(f"Field-coordinate progressions found: {len(progression_df)}")
print(f"Saved to {OUTPUT_CSV}")

if len(progression_df) > 0:
    print("\nLargest field-coordinate progressions:")
    print(
        progression_df.sort_values("field_distance_yards", ascending=False)
        .head(15)[[
            "sequence_id",
            "start_time",
            "end_time",
            "duration_seconds",
            "num_detections",
            "field_distance_yards",
            "field_dx_yards",
            "field_dy_yards",
            "avg_speed_yards_per_second",
            "movement_shape",
            "avg_confidence",
        ]]
        .to_string(index=False)
    )