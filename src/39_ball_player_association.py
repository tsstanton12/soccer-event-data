import argparse
from pathlib import Path

import pandas as pd


def associate_ball_to_players(ball_csv, player_csv, output_csv, max_distance_px=150):
    ball_csv = Path(ball_csv)
    player_csv = Path(player_csv)
    output_csv = Path(output_csv)

    ball = pd.read_csv(ball_csv)
    players = pd.read_csv(player_csv)

    ball_required = ["frame", "center_x", "center_y", "source"]
    player_required = ["frame", "center_x", "center_y"]

    missing_ball = [c for c in ball_required if c not in ball.columns]
    missing_players = [c for c in player_required if c not in players.columns]

    if missing_ball:
        raise ValueError(f"Ball CSV missing required columns: {missing_ball}")
    if missing_players:
        raise ValueError(f"Player CSV missing required columns: {missing_players}")

    ball["frame"] = ball["frame"].astype(int)
    players["frame"] = players["frame"].astype(int)

    # Add player_id if your player tracker does not already have one
    if "player_id" not in players.columns:
        players["player_id"] = players.groupby("frame").cumcount() + 1

    players_by_frame = {
        frame: group.to_dict("records")
        for frame, group in players.groupby("frame")
    }

    output_rows = []

    for _, b in ball.iterrows():
        frame = int(b["frame"])
        bx = float(b["center_x"])
        by = float(b["center_y"])

        frame_players = players_by_frame.get(frame, [])

        if not frame_players:
            output_rows.append({
                **b.to_dict(),
                "nearest_player_id": None,
                "nearest_player_distance_px": None,
                "nearest_player_x": None,
                "nearest_player_y": None,
                "association_status": "no_players_in_frame",
            })
            continue

        best_player = None
        best_distance = None

        for p in frame_players:
            px = float(p["center_x"])
            py = float(p["center_y"])

            distance = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_player = p

        if best_distance <= max_distance_px:
            status = "associated"
        else:
            status = "too_far"

        output_rows.append({
            **b.to_dict(),
            "nearest_player_id": best_player.get("player_id"),
            "nearest_player_distance_px": best_distance,
            "nearest_player_x": best_player.get("center_x"),
            "nearest_player_y": best_player.get("center_y"),
            "association_status": status,
        })

    out = pd.DataFrame(output_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    total = len(out)
    associated = (out["association_status"] == "associated").sum()
    too_far = (out["association_status"] == "too_far").sum()
    no_players = (out["association_status"] == "no_players_in_frame").sum()

    print("BALL-PLAYER ASSOCIATION COMPLETE")
    print("--------------------------------")
    print(f"Ball CSV: {ball_csv}")
    print(f"Player CSV: {player_csv}")
    print(f"Output CSV: {output_csv}")
    print("")
    print(f"Total ball rows: {total}")
    print(f"Associated: {associated}")
    print(f"Too far: {too_far}")
    print(f"No players in frame: {no_players}")
    print("")
    print(f"Association rate: {associated / total * 100:.1f}%")
    print(f"Max association distance: {max_distance_px}px")


def main():
    parser = argparse.ArgumentParser(
        description="Associate each ball position with the nearest detected player."
    )

    parser.add_argument("ball_csv", help="Interpolated ball path CSV")
    parser.add_argument("player_csv", help="Player detections/tracks CSV")
    parser.add_argument("--output", "-o", required=True, help="Output association CSV")
    parser.add_argument(
        "--max-distance-px",
        type=float,
        default=150,
        help="Max distance for a valid ball-player association. Default: 150px",
    )

    args = parser.parse_args()

    associate_ball_to_players(
        ball_csv=args.ball_csv,
        player_csv=args.player_csv,
        output_csv=args.output,
        max_distance_px=args.max_distance_px,
    )


if __name__ == "__main__":
    main()