import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/ball_progressions.csv"
OUTPUT_CSV = "outputs/progression_review_sheet.csv"

progressions = pd.read_csv(INPUT_CSV)

review = progressions.copy()

# Add human-review columns
review["review_status"] = ""
review["event_label"] = ""
review["notes"] = ""

# Make times easier to read
review["start_time_mmss"] = review["start_time"].apply(
    lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
)

review["end_time_mmss"] = review["end_time"].apply(
    lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
)

# Reorder columns
columns = [
    "sequence_id",
    "review_status",
    "event_label",
    "notes",
    "start_time_mmss",
    "end_time_mmss",
    "duration_seconds",
    "pixel_distance",
    "avg_speed_pixels_per_second",
    "num_detections",
    "avg_confidence",
    "start_frame",
    "end_frame",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
]

review = review[columns]

Path("outputs").mkdir(exist_ok=True)
review.to_csv(OUTPUT_CSV, index=False)

print(f"Saved review sheet to {OUTPUT_CSV}")
print("\nSuggested event_label values:")
print("- long_ball")
print("- switch")
print("- clearance")
print("- cross")
print("- shot")
print("- counterattack_candidate")
print("- bad_tracking")
print("- ignore")