import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = "outputs/conservative_ball_path.csv"
OUTPUT_CSV = "outputs/ball_progressions.csv"

# Settings
MAX_GAP_SECONDS_WITHIN_SEQUENCE = 1.5
MIN_SEQUENCE_DURATION = 1.0
MAX_SEQUENCE_DURATION = 12.0
MIN_PIXEL_DISTANCE = 180
MIN_AVG_SPEED = 40

ball = pd.read_csv(INPUT_CSV)
ball = ball.sort_values("time_seconds").copy()

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

    start_x = start["center_x"]
    start_y = start["center_y"]
    end_x = end["center_x"]
    end_y = end["center_y"]

    dx = end_x - start_x
    dy = end_y - start_y

    pixel_distance = np.sqrt(dx**2 + dy**2)
    avg_speed = pixel_distance / duration if duration > 0 else 0

    if pixel_distance < MIN_PIXEL_DISTANCE:
        continue

    if avg_speed < MIN_AVG_SPEED:
        continue

    if abs(dx) > abs(dy):
        direction = "horizontal"
    else:
        direction = "vertical"

    progressions.append({
        "sequence_id": int(sequence_id),
        "start_frame": int(start["frame"]),
        "end_frame": int(end["frame"]),
        "start_time": start["time_seconds"],
        "end_time": end["time_seconds"],
        "duration_seconds": duration,
        "num_detections": len(group),
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "dx": dx,
        "dy": dy,
        "pixel_distance": pixel_distance,
        "avg_speed_pixels_per_second": avg_speed,
        "avg_confidence": group["confidence"].mean(),
    })

progression_df = pd.DataFrame(progressions)

Path("outputs").mkdir(exist_ok=True)
progression_df.to_csv(OUTPUT_CSV, index=False)

print("\nBALL PROGRESSION REPORT")
print("-----------------------")
print(f"Input detections: {len(ball)}")
print(f"Progressions found: {len(progression_df)}")
print(f"Saved to {OUTPUT_CSV}")

if len(progression_df) > 0:
    print("\nLargest progressions:")
    print(
        progression_df.sort_values("pixel_distance", ascending=False)
        .head(15)[[
            "sequence_id",
            "start_time",
            "end_time",
            "duration_seconds",
            "num_detections",
            "pixel_distance",
            "avg_speed_pixels_per_second",
            "avg_confidence",
        ]]
        .to_string(index=False)
    )