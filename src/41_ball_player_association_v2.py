import argparse
from pathlib import Path

import pandas as pd


def parse_polygon(polygon_string):
    if not polygon_string:
        return None

    points = []
    pairs = polygon_string.split(";")

    for pair in pairs:
        x, y = pair.split(",")
        points.append((float(x), float(y)))

    return points


def point_inside_polygon(x, y, polygon):
    if polygon is None:
        return True

    inside = False
    n = len(polygon)

    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x

                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def associate_ball_to_on_field_players(
    ball_csv,
    player_csv,
    output_csv,
    fps=30,
    strict_field_polygon=None,
    max_distance_px=150,
    control_distance_px=80,
    controlled_speed_px_per_second=500,
    transit_speed_px_per_second=700,
):
    ball_csv = Path(ball_csv)
    player_csv = Path(player_csv)
    output_csv = Path(output_csv)

    ball = pd.read_csv(ball_csv)
    players = pd.read_csv(player_csv)

    ball["frame"] = ball["frame"].astype(int)
    players["frame"] = players["frame"].astype(int)

    # Player foot point: better than center for soccer possession
    players["foot_x"] = (players["x1"] + players["x2"]) / 2
    players["foot_y"] = players["y2"]

    if "field_zone" in players.columns:
        players["on_field"] = players["field_zone"].isin(["strict", "tolerant"])
    else:
        players["on_field"] = players.apply(
            lambda r: point_inside_polygon(
                r["foot_x"], r["foot_y"], strict_field_polygon
            ),
            axis=1,
        )
        players["field_zone"] = players["on_field"].map(
            {True: "strict", False: "off_field"}
        )

    eligible_players = players[players["on_field"] == True].copy()

    players_by_frame = {
        frame: group.to_dict("records")
        for frame, group in eligible_players.groupby("frame")
    }

    ball = ball.sort_values("frame").reset_index(drop=True)

    ball["prev_center_x"] = ball["center_x"].shift(1)
    ball["prev_center_y"] = ball["center_y"].shift(1)
    ball["prev_frame"] = ball["frame"].shift(1)

    ball["dx"] = ball["center_x"] - ball["prev_center_x"]
    ball["dy"] = ball["center_y"] - ball["prev_center_y"]
    ball["dt_seconds"] = (ball["frame"] - ball["prev_frame"]) / fps
    ball["ball_speed_px_per_second"] = (
        ((ball["dx"] ** 2 + ball["dy"] ** 2) ** 0.5) / ball["dt_seconds"]
    )

    output_rows = []

    for _, b in ball.iterrows():
        frame = int(b["frame"])
        bx = float(b["center_x"])
        by = float(b["center_y"])
        speed = b["ball_speed_px_per_second"]

        frame_players = players_by_frame.get(frame, [])

        if not frame_players:
            row = b.to_dict()
            row.update({
                "nearest_player_id": None,
                "nearest_player_distance_px": None,
                "nearest_player_foot_x": None,
                "nearest_player_foot_y": None,
                "nearest_player_field_zone": None,
                "association_status": "no_on_field_players",
                "ball_state": "unknown",
            })
            output_rows.append(row)
            continue

        best_player = None
        best_distance = None

        strict_players = [p for p in frame_players if p["field_zone"] == "strict"]
        strict_distances = [
            ((bx - float(p["foot_x"])) ** 2 + (by - float(p["foot_y"])) ** 2) ** 0.5
            for p in strict_players
        ]
        candidates = (
            strict_players
            if strict_distances and min(strict_distances) <= max_distance_px
            else frame_players
        )

        for p in candidates:
            px = float(p["foot_x"])
            py = float(p["foot_y"])

            distance = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_player = p

        if best_distance > max_distance_px:
            association_status = "too_far"
            ball_state = "loose_or_unclear"

        elif pd.notna(speed) and speed >= transit_speed_px_per_second:
            association_status = "near_player_but_fast"
            ball_state = "in_transit"

        elif best_distance <= control_distance_px and (
            pd.isna(speed) or speed <= controlled_speed_px_per_second
        ):
            association_status = "associated"
            ball_state = "controlled"

        else:
            association_status = "near_player_unclear"
            ball_state = "loose_or_unclear"

        row = b.to_dict()
        row.update({
            "nearest_player_id": best_player.get("player_id"),
            "nearest_player_distance_px": best_distance,
            "nearest_player_foot_x": best_player.get("foot_x"),
            "nearest_player_foot_y": best_player.get("foot_y"),
            "nearest_player_field_zone": best_player.get("field_zone"),
            "association_status": association_status,
            "ball_state": ball_state,
        })

        output_rows.append(row)

    out = pd.DataFrame(output_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("BALL-PLAYER ASSOCIATION V2 COMPLETE")
    print("-----------------------------------")
    print(f"Ball CSV: {ball_csv}")
    print(f"Player CSV: {player_csv}")
    print(f"Output CSV: {output_csv}")
    print("")
    print(f"Total ball rows: {len(out)}")
    print(f"Eligible on-field player detections: {len(eligible_players)}")
    print(f"Filtered off-field player detections: {len(players) - len(eligible_players)}")
    print("")
    print("Association status:")
    print(out["association_status"].value_counts(dropna=False))
    print("")
    print("Ball state:")
    print(out["ball_state"].value_counts(dropna=False))


def main():
    parser = argparse.ArgumentParser(
        description="Associate ball to nearest on-field player using foot point and ball-state logic."
    )

    parser.add_argument("ball_csv")
    parser.add_argument("player_csv")
    parser.add_argument("--output", "-o", required=True)

    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--max-distance-px", type=float, default=150)
    parser.add_argument("--control-distance-px", type=float, default=80)
    parser.add_argument("--controlled-speed", type=float, default=500)
    parser.add_argument("--transit-speed", type=float, default=700)

    parser.add_argument(
        "--strict-field-polygon",
        default=None,
        help='Strict field polygon as "x1,y1;x2,y2;x3,y3;x4,y4". Uses player foot point.',
    )
    parser.add_argument(
        "--field-polygon",
        default=None,
        help="Deprecated alias for --strict-field-polygon.",
    )

    args = parser.parse_args()

    if args.strict_field_polygon and args.field_polygon:
        parser.error("Use only one of --strict-field-polygon and --field-polygon.")

    strict_polygon = parse_polygon(args.strict_field_polygon or args.field_polygon)

    associate_ball_to_on_field_players(
        ball_csv=args.ball_csv,
        player_csv=args.player_csv,
        output_csv=args.output,
        fps=args.fps,
        strict_field_polygon=strict_polygon,
        max_distance_px=args.max_distance_px,
        control_distance_px=args.control_distance_px,
        controlled_speed_px_per_second=args.controlled_speed,
        transit_speed_px_per_second=args.transit_speed,
    )


if __name__ == "__main__":
    main()
