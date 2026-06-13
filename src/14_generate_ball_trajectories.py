import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = "outputs/filtered_ball_detections.csv"
OUTPUT_CSV = "outputs/ball_trajectories.csv"

# A new segment starts if the ball disappears for too long
MAX_FRAME_GAP_WITHIN_SEGMENT = 15  # about 0.5 seconds at 30 fps

# Keep only meaningful short movement chunks
MIN_SEGMENT_DURATION = 0.5
MAX_SEGMENT_DURATION = 5.0
MIN_PIXEL_DISTANCE = 40

df = pd.read_csv(INPUT_CSV)
df = df.sort_values("frame").copy()

df["frame_gap"] = df["frame"].diff().fillna(0)
df["new_segment"] = df["frame_gap"] > MAX_FRAME_GAP_WITHIN_SEGMENT
df["segment_id"] = df["new_segment"].cumsum() + 1

segments = []

for segment_id, group in df.groupby("segment_id"):
    group = group.sort_values("frame")

    start = group.iloc[0]
    end = group.iloc[-1]

    start_time = start["time_seconds"]
    end_time = end["time_seconds"]
    duration = end_time - start_time

    if duration < MIN_SEGMENT_DURATION:
        continue

    if duration > MAX_SEGMENT_DURATION:
        continue

    start_x = start["center_x"]
    start_y = start["center_y"]
    end_x = end["center_x"]
    end_y = end["center_y"]

    pixel_distance = np.sqrt(
        (end_x - start_x) ** 2 +
        (end_y - start_y) ** 2
    )

    if pixel_distance < MIN_PIXEL_DISTANCE:
        continue

    avg_speed_pixels_per_second = pixel_distance / duration if duration > 0 else 0

    segments.append({
        "segment_id": int(segment_id),
        "start_frame": int(start["frame"]),
        "end_frame": int(end["frame"]),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "num_detections": len(group),
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "pixel_distance": pixel_distance,
        "avg_speed_pixels_per_second": avg_speed_pixels_per_second,
        "start_confidence": start["confidence"],
        "end_confidence": end["confidence"],
        "avg_confidence": group["confidence"].mean(),
    })

trajectory_df = pd.DataFrame(segments)

Path("outputs").mkdir(exist_ok=True)
trajectory_df.to_csv(OUTPUT_CSV, index=False)

print("\nBALL TRAJECTORY REPORT")
print("----------------------")
print(f"Input ball detections: {len(df)}")
print(f"Trajectory segments created: {len(trajectory_df)}")
print(f"Saved to {OUTPUT_CSV}")

if len(trajectory_df) > 0:
    print("\nLargest movement segments:")
    print(
        trajectory_df.sort_values("pixel_distance", ascending=False)
        .head(10)[[
            "segment_id",
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