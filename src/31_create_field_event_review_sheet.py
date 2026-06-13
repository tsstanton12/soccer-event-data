import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/field_event_candidates.csv"
OUTPUT_CSV = "outputs/field_event_review_sheet.csv"

df = pd.read_csv(INPUT_CSV)

df["start_time_mmss"] = df["start_time"].apply(
    lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
)

df["end_time_mmss"] = df["end_time"].apply(
    lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
)

# Keep review columns near front
columns = [
    "sequence_id",
    "event_candidate",
    "review_status",
    "final_event_label",
    "notes",
    "start_time_mmss",
    "end_time_mmss",
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

print(f"Saved field event review sheet to {OUTPUT_CSV}")

print("\nSuggested review_status values:")
print("- good")
print("- bad")
print("- unsure")

print("\nSuggested final_event_label values:")
print("- simple_pass")
print("- switch")
print("- direct_progression")
print("- diagonal_progression")
print("- clearance")
print("- cross")
print("- shot")
print("- counterattack_candidate")
print("- bad_tracking")
print("- ignore")