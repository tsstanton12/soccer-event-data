# 43_filter_on_field_players.py

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.path import Path as MplPath


def parse_polygon(poly_string):
    """
    Parse polygon string like:
    "100,120;1800,120;1850,1000;80,1000"
    """
    points = []
    for pair in poly_string.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))

    if len(points) < 3:
        raise ValueError("Field polygon must have at least 3 points.")

    return points


def classify_points(df, strict_polygon, tolerant_polygon):
    points = df[["foot_x", "foot_y"]].to_numpy()
    strict = MplPath(strict_polygon).contains_points(points)
    tolerant = MplPath(tolerant_polygon).contains_points(points)

    zones = pd.Series("off_field", index=df.index)
    zones.loc[df.index[strict]] = "strict"
    zones.loc[df.index[~strict & tolerant]] = "tolerant"
    return zones


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--player_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--field_masks_csv",
        default=None,
        help="Frame-specific polygons from Script 45. Uses the nearest previous mask.",
    )

    parser.add_argument(
        "--strict_field_polygon",
        default=None,
        help='Optional static strict polygon as "x,y;x,y;x,y;x,y"',
    )
    parser.add_argument(
        "--tolerant_field_polygon",
        default=None,
        help='Optional static tolerant polygon as "x,y;x,y;x,y;x,y"',
    )
    parser.add_argument(
        "--field_polygon",
        default=None,
        help="Deprecated alias for --tolerant_field_polygon.",
    )

    args = parser.parse_args()

    player_csv = Path(args.player_csv)
    output_csv = Path(args.output_csv)
    if args.tolerant_field_polygon and args.field_polygon:
        parser.error("Use only one of --tolerant_field_polygon and --field_polygon.")

    tolerant_polygon_string = args.tolerant_field_polygon or args.field_polygon
    if not args.field_masks_csv and not tolerant_polygon_string:
        parser.error("--field_masks_csv or --tolerant_field_polygon is required.")

    if not player_csv.exists():
        raise FileNotFoundError(f"Missing player CSV: {player_csv}")

    df = pd.read_csv(player_csv)

    required_cols = ["x1", "y1", "x2", "y2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in player CSV: {missing}")

    df["foot_x"] = (df["x1"] + df["x2"]) / 2
    df["foot_y"] = df["y2"]

    df["field_zone"] = "off_field"

    if args.field_masks_csv:
        masks = pd.read_csv(args.field_masks_csv)
        required_mask_cols = ["frame"]
        missing = [c for c in required_mask_cols if c not in masks.columns]
        if missing:
            raise ValueError(f"Missing columns in field masks CSV: {missing}")

        if "strict_field_polygon" not in masks.columns:
            if "field_polygon" not in masks.columns:
                raise ValueError(
                    "Field masks CSV must contain strict_field_polygon or field_polygon."
                )
            masks["strict_field_polygon"] = masks["field_polygon"]

        if "tolerant_field_polygon" not in masks.columns:
            masks["tolerant_field_polygon"] = masks["strict_field_polygon"]

        masks = masks.dropna(subset=["frame", "strict_field_polygon"]).copy()
        masks = masks[masks["strict_field_polygon"].astype(str).str.len() > 0]
        masks["frame"] = masks["frame"].astype(int)
        masks = masks.sort_values("frame").drop_duplicates("frame", keep="last")

        for frame, frame_df in df.groupby("frame"):
            previous = masks[masks["frame"] <= int(frame)]
            if previous.empty:
                continue
            mask_row = previous.iloc[-1]
            df.loc[frame_df.index, "field_zone"] = classify_points(
                df.loc[frame_df.index],
                parse_polygon(mask_row["strict_field_polygon"]),
                parse_polygon(mask_row["tolerant_field_polygon"]),
            )
    else:
        tolerant_field_polygon = parse_polygon(tolerant_polygon_string)
        strict_field_polygon = parse_polygon(
            args.strict_field_polygon or tolerant_polygon_string
        )
        df["field_zone"] = classify_points(
            df, strict_field_polygon, tolerant_field_polygon
        )

    df["is_on_field"] = df["field_zone"].isin(["strict", "tolerant"])
    on_field_df = df[df["is_on_field"]].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    on_field_df.to_csv(output_csv, index=False)

    print("FIELD FILTER REPORT")
    print("-------------------")
    print(f"Input player detections: {len(df)}")
    print(f"On-field player detections: {len(on_field_df)}")
    print(f"Strict player detections: {(df['field_zone'] == 'strict').sum()}")
    print(f"Tolerant-only player detections: {(df['field_zone'] == 'tolerant').sum()}")
    print(f"Filtered off-field detections: {len(df) - len(on_field_df)}")
    print(f"Field masks: {args.field_masks_csv or 'static polygons'}")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()
