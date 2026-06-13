import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CALIBRATION_JSON = "outputs/field_calibration_points.json"
INPUT_CSV = "outputs/conservative_ball_path.csv"
OUTPUT_CSV = "outputs/conservative_ball_path_field_coords.csv"

with open(CALIBRATION_JSON, "r") as f:
    calibration = json.load(f)

points = calibration["points"]

if len(points) < 4:
    raise ValueError("Need at least 4 calibration points.")

pixel_points = np.array(
    [p["pixel"] for p in points],
    dtype=np.float32
)

field_points = np.array(
    [p["field"] for p in points],
    dtype=np.float32
)

# Compute pixel → field transform
homography_matrix, mask = cv2.findHomography(
    pixel_points,
    field_points,
    method=0
)

if homography_matrix is None:
    raise RuntimeError("Could not compute field calibration transform.")

ball = pd.read_csv(INPUT_CSV)

pixel_ball_points = ball[["center_x", "center_y"]].to_numpy(dtype=np.float32)
pixel_ball_points = pixel_ball_points.reshape(-1, 1, 2)

field_ball_points = cv2.perspectiveTransform(
    pixel_ball_points,
    homography_matrix
).reshape(-1, 2)

ball["field_x_yards"] = field_ball_points[:, 0]
ball["field_y_yards"] = field_ball_points[:, 1]

# Optional cleanup: mark whether estimated point is inside normal field bounds
ball["inside_field_bounds"] = (
    (ball["field_x_yards"] >= 0) &
    (ball["field_x_yards"] <= 120) &
    (ball["field_y_yards"] >= 0) &
    (ball["field_y_yards"] <= 75)
)

Path("outputs").mkdir(exist_ok=True)
ball.to_csv(OUTPUT_CSV, index=False)

print("\nFIELD CALIBRATION APPLIED")
print("-------------------------")
print(f"Calibration points used: {len(points)}")
print(f"Input ball detections: {len(ball)}")
print(f"Saved to {OUTPUT_CSV}")

print("\nField coordinate summary:")
print(ball[["field_x_yards", "field_y_yards", "inside_field_bounds"]].describe())

outside = (~ball["inside_field_bounds"]).sum()
print(f"\nDetections outside 0–120 x 0–75 field bounds: {outside}")