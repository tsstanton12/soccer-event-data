import pandas as pd

PLAYER_CSV = "outputs/detections.csv"
BALL_CSV = "outputs/filtered_ball_detections.csv"

players = pd.read_csv(PLAYER_CSV)
ball = pd.read_csv(BALL_CSV)

players = players[players["object_type"] == "person"].copy()

video_duration = max(
    players["time_seconds"].max(),
    ball["time_seconds"].max()
)

player_frames = players["frame"].nunique()
ball_frames = ball["frame"].nunique()

avg_players_per_frame = players.groupby("frame").size().mean()

ball = ball.sort_values("frame")
ball["frame_gap"] = ball["frame"].diff()

avg_ball_gap = ball["frame_gap"].mean()
max_ball_gap = ball["frame_gap"].max()

ball_detections_per_second = len(ball) / video_duration
player_detections_per_second = len(players) / video_duration

print("\nTRACKING QUALITY REPORT")
print("-----------------------")
print(f"Video duration: {video_duration:.1f} seconds")
print(f"Total player detections: {len(players)}")
print(f"Total ball detections: {len(ball)}")
print(f"Frames with player detections: {player_frames}")
print(f"Frames with ball detections: {ball_frames}")
print(f"Average players detected per sampled frame: {avg_players_per_frame:.1f}")
print(f"Player detections per second: {player_detections_per_second:.1f}")
print(f"Ball detections per second: {ball_detections_per_second:.2f}")
print(f"Average frame gap between ball detections: {avg_ball_gap:.1f}")
print(f"Largest frame gap between ball detections: {max_ball_gap:.1f}")

print("\nROUGH INTERPRETATION")
print("--------------------")

if avg_players_per_frame >= 14:
    print("Player detection: GOOD")
elif avg_players_per_frame >= 8:
    print("Player detection: USABLE")
else:
    print("Player detection: WEAK")

if ball_detections_per_second >= 2:
    print("Ball detection volume: GOOD")
elif ball_detections_per_second >= 0.75:
    print("Ball detection volume: USABLE")
else:
    print("Ball detection volume: WEAK")

if max_ball_gap <= 60:
    print("Ball continuity: GOOD")
elif max_ball_gap <= 150:
    print("Ball continuity: USABLE WITH INTERPOLATION")
else:
    print("Ball continuity: TOO GAPPY FOR FULL EVENT DATA")