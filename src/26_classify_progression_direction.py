import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/ball_progressions.csv"
OUTPUT_CSV = "outputs/ball_progressions_classified.csv"

# For now, this is pixel-based.
# Later we will replace this with true field coordinates.
progressions = pd.read_csv(INPUT_CSV)

classified = progressions.copy()

def classify_direction(row):
    dx = row["dx"]
    dy = row["dy"]

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx > abs_dy * 1.5:
        if dx > 0:
            return "right"
        else:
            return "left"

    if abs_dy > abs_dx * 1.5:
        if dy > 0:
            return "down"
        else:
            return "up"

    return "diagonal"

def classify_shape(row):
    dx = row["dx"]
    dy = row["dy"]

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx > abs_dy * 1.8:
        return "mostly_horizontal"
    elif abs_dy > abs_dx * 1.8:
        return "mostly_vertical"
    else:
        return "diagonal"

def classify_length(row):
    distance = row["pixel_distance"]

    if distance >= 400:
        return "long"
    elif distance >= 250:
        return "medium"
    else:
        return "short"

def classify_speed(row):
    speed = row["avg_speed_pixels_per_second"]

    if speed >= 120:
        return "fast"
    elif speed >= 70:
        return "medium"
    else:
        return "slow"

classified["movement_direction"] = classified.apply(classify_direction, axis=1)
classified["movement_shape"] = classified.apply(classify_shape, axis=1)
classified["movement_length"] = classified.apply(classify_length, axis=1)
classified["movement_speed"] = classified.apply(classify_speed, axis=1)

# Basic candidate labels
def suggest_event(row):
    length = row["movement_length"]
    speed = row["movement_speed"]
    shape = row["movement_shape"]

    if length == "long" and speed in ["fast", "medium"]:
        return "direct_ball_movement_candidate"

    if length in ["medium", "long"] and shape == "mostly_horizontal":
        return "switch_or_lateral_progression_candidate"

    if length == "medium":
        return "pass_candidate"

    return "minor_progression"

classified["suggested_event_type"] = classified.apply(suggest_event, axis=1)

Path("outputs").mkdir(exist_ok=True)
classified.to_csv(OUTPUT_CSV, index=False)

print("\nCLASSIFIED PROGRESSION REPORT")
print("-----------------------------")
print(f"Input progressions: {len(progressions)}")
print(f"Saved to {OUTPUT_CSV}")

print("\nSuggested event counts:")
print(classified["suggested_event_type"].value_counts().to_string())

print("\nClassified progressions:")
print(
    classified[[
        "sequence_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "pixel_distance",
        "avg_speed_pixels_per_second",
        "movement_direction",
        "movement_shape",
        "movement_length",
        "movement_speed",
        "suggested_event_type",
    ]]
    .to_string(index=False)
)