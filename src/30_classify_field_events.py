import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/field_coordinate_progressions.csv"
OUTPUT_CSV = "outputs/field_event_candidates.csv"

df = pd.read_csv(INPUT_CSV)

def classify_event(row):
    distance = row["field_distance_yards"]
    dx = row["field_dx_yards"]
    dy = row["field_dy_yards"]
    speed = row["avg_speed_yards_per_second"]
    duration = row["duration_seconds"]

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Widthwise movement: possible switch / lateral circulation
    if abs_dy > abs_dx * 1.5 and distance >= 25:
        if speed >= 8:
            return "switch_candidate"
        else:
            return "lateral_circulation_candidate"

    # Lengthwise movement: direct progression / clearance / transition
    if abs_dx > abs_dy * 1.5 and distance >= 25:
        if speed >= 8 and duration <= 6:
            return "direct_progression_candidate"
        elif speed < 8:
            return "buildup_progression_candidate"

    # Big diagonal movement
    if distance >= 30:
        if speed >= 8:
            return "diagonal_direct_ball_candidate"
        else:
            return "diagonal_progression_candidate"

    # Medium pass
    if distance >= 15:
        return "simple_pass_candidate"

    return "minor_movement"

def classify_length(row):
    d = row["field_distance_yards"]
    if d >= 40:
        return "very_long"
    elif d >= 30:
        return "long"
    elif d >= 20:
        return "medium"
    else:
        return "short"

def classify_speed(row):
    s = row["avg_speed_yards_per_second"]
    if s >= 12:
        return "fast"
    elif s >= 7:
        return "medium"
    else:
        return "slow"

df["event_candidate"] = df.apply(classify_event, axis=1)
df["length_bucket"] = df.apply(classify_length, axis=1)
df["speed_bucket"] = df.apply(classify_speed, axis=1)

# Add blank human review columns
df["review_status"] = ""
df["final_event_label"] = ""
df["notes"] = ""

columns = [
    "sequence_id",
    "event_candidate",
    "review_status",
    "final_event_label",
    "notes",
    "start_time",
    "end_time",
    "duration_seconds",
    "field_distance_yards",
    "field_dx_yards",
    "field_dy_yards",
    "avg_speed_yards_per_second",
    "movement_shape",
    "length_bucket",
    "speed_bucket",
    "num_detections",
    "avg_confidence",
    "start_frame",
    "end_frame",
    "start_field_x",
    "start_field_y",
    "end_field_x",
    "end_field_y",
]

df = df[columns]

Path("outputs").mkdir(exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("\nFIELD EVENT CANDIDATE REPORT")
print("----------------------------")
print(f"Input field progressions: {len(df)}")
print(f"Saved to {OUTPUT_CSV}")

print("\nEvent candidate counts:")
print(df["event_candidate"].value_counts().to_string())

print("\nEvent candidates:")
print(
    df[[
        "sequence_id",
        "event_candidate",
        "start_time",
        "end_time",
        "duration_seconds",
        "field_distance_yards",
        "field_dx_yards",
        "field_dy_yards",
        "avg_speed_yards_per_second",
        "length_bucket",
        "speed_bucket",
    ]]
    .to_string(index=False)
)