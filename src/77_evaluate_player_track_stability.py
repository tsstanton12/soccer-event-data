#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def percentile(series, value):
    if series.empty:
        return 0.0
    return float(series.quantile(value))


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def estimate_fps(rows):
    frames = rows[["frame", "time_seconds"]].drop_duplicates().sort_values("frame")
    if len(frames) < 2:
        return 30.0
    duration = float(frames["time_seconds"].max() - frames["time_seconds"].min())
    frame_span = int(frames["frame"].max() - frames["frame"].min())
    if duration <= 0 or frame_span <= 0:
        return 30.0
    return frame_span / duration


def add_geometry(rows):
    rows = rows.copy()
    if "foot_x" not in rows.columns:
        rows["foot_x"] = (rows["x1"] + rows["x2"]) / 2
    if "foot_y" not in rows.columns:
        rows["foot_y"] = rows["y2"]
    rows["box_width_px"] = rows["x2"] - rows["x1"]
    rows["box_height_px"] = rows["y2"] - rows["y1"]
    return rows


def summarize_tracks(tracks, association=None, min_stable_seconds=2.0, jump_px_per_frame=35.0):
    tracks = add_geometry(tracks)
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)
    fps = estimate_fps(tracks)
    min_stable_frames = max(2, int(round(min_stable_seconds * fps)))

    track_rows = []
    for track_id, group in tracks.sort_values(["track_id", "frame"]).groupby("track_id"):
        group = group.sort_values("frame")
        frame_diffs = group["frame"].diff()
        foot_dx = group["foot_x"].diff()
        foot_dy = group["foot_y"].diff()
        foot_dist = (foot_dx * foot_dx + foot_dy * foot_dy) ** 0.5
        per_frame = foot_dist / frame_diffs.clip(lower=1)
        gaps = frame_diffs[frame_diffs > 1]
        big_jumps = per_frame[per_frame > jump_px_per_frame]

        track_rows.append(
            {
                "track_id": int(track_id),
                "detections": int(len(group)),
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "span_frames": int(group["frame"].max() - group["frame"].min() + 1),
                "duration_seconds": float(
                    group["time_seconds"].max() - group["time_seconds"].min()
                )
                if "time_seconds" in group
                else None,
                "coverage_rate": safe_rate(
                    len(group), int(group["frame"].max() - group["frame"].min() + 1)
                ),
                "mean_confidence": float(group["confidence"].mean()),
                "median_confidence": float(group["confidence"].median()),
                "median_box_height_px": float(group["box_height_px"].median()),
                "median_box_width_px": float(group["box_width_px"].median()),
                "strict_rows": int((group.get("field_zone", "") == "strict").sum()),
                "tolerant_rows": int((group.get("field_zone", "") == "tolerant").sum()),
                "off_field_rows": int((group.get("field_zone", "") == "off_field").sum()),
                "gap_count": int(len(gaps)),
                "max_gap_frames": int(gaps.max()) if len(gaps) else 0,
                "big_jump_count": int(len(big_jumps)),
                "max_step_px_per_frame": float(per_frame.max()) if len(per_frame.dropna()) else 0.0,
                "median_step_px_per_frame": float(per_frame.median())
                if len(per_frame.dropna())
                else 0.0,
                "new_rows": int((group.get("track_status", "") == "new").sum()),
                "matched_rows": int((group.get("track_status", "") == "matched").sum()),
                "is_stable_2s": bool(len(group) >= min_stable_frames),
            }
        )

    summary = pd.DataFrame(track_rows)

    if association is not None and not association.empty:
        association = association.copy()
        if "nearest_track_id" in association.columns:
            nearest_col = "nearest_track_id"
        elif "nearest_player_id" in association.columns:
            nearest_col = "nearest_player_id"
        else:
            nearest_col = None

        if nearest_col:
            nearest = association[pd.notna(association[nearest_col])].copy()
            nearest["_track_id"] = nearest[nearest_col].astype(float).astype(int)
            nearest_counts = nearest.groupby("_track_id").size().rename("nearest_rows")
            controlled_counts = (
                nearest[nearest.get("ball_state", "") == "controlled"]
                .groupby("_track_id")
                .size()
                .rename("controlled_nearest_rows")
            )
            if "possession_player_id" in nearest.columns:
                possession_counts = (
                    nearest[pd.notna(nearest["possession_player_id"])]
                    .assign(
                        _possession_track=lambda df: df["possession_player_id"]
                        .astype(float)
                        .astype(int)
                    )
                    .groupby("_possession_track")
                    .size()
                    .rename("possession_rows")
                )
            else:
                possession_counts = pd.Series(dtype="int64", name="possession_rows")
            summary = summary.merge(
                nearest_counts,
                left_on="track_id",
                right_index=True,
                how="left",
            )
            summary = summary.merge(
                controlled_counts,
                left_on="track_id",
                right_index=True,
                how="left",
            )
            summary = summary.merge(
                possession_counts,
                left_on="track_id",
                right_index=True,
                how="left",
            )

    for col in ["nearest_rows", "controlled_nearest_rows", "possession_rows"]:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    summary["stability_flag"] = "ok"
    summary.loc[summary["detections"] < min_stable_frames, "stability_flag"] = "short_track"
    summary.loc[
        (summary["is_stable_2s"]) & (summary["coverage_rate"] < 0.50),
        "stability_flag",
    ] = "gappy_track"
    summary.loc[
        (summary["is_stable_2s"]) & (summary["big_jump_count"] >= 3),
        "stability_flag",
    ] = "jumpy_track"
    summary.loc[
        (summary["is_stable_2s"])
        & (summary["coverage_rate"] >= 0.50)
        & (summary["big_jump_count"] < 3),
        "stability_flag",
    ] = "stable"

    return summary, fps, min_stable_frames


def build_report(track_summary, fps, min_stable_frames, output_dir, label):
    lines = [
        f"# Player Track Stability Report: {label}",
        "",
        f"- Estimated FPS: {fps:.2f}",
        f"- Stable-track threshold: {min_stable_frames} detections",
        f"- Tracks: {len(track_summary)}",
        f"- Stable tracks: {int((track_summary['stability_flag'] == 'stable').sum())}",
        f"- Short tracks: {int((track_summary['stability_flag'] == 'short_track').sum())}",
        f"- Gappy tracks: {int((track_summary['stability_flag'] == 'gappy_track').sum())}",
        f"- Jumpy tracks: {int((track_summary['stability_flag'] == 'jumpy_track').sum())}",
        "",
        "## Track Lengths",
        "",
        f"- Median detections/track: {track_summary['detections'].median():.1f}",
        f"- 75th percentile: {percentile(track_summary['detections'], 0.75):.1f}",
        f"- 90th percentile: {percentile(track_summary['detections'], 0.90):.1f}",
        f"- Max detections: {int(track_summary['detections'].max()) if len(track_summary) else 0}",
        "",
        "## Possession-Relevant Tracks",
        "",
        "These are the tracks most often nearest to the ball or assigned possession.",
        "",
    ]

    top = track_summary.sort_values(
        ["possession_rows", "controlled_nearest_rows", "nearest_rows", "detections"],
        ascending=False,
    ).head(20)
    if top.empty:
        lines.append("No association data was supplied.")
    else:
        lines.append(
            "| track_id | flag | detections | coverage | big jumps | nearest rows | controlled nearest | possession rows |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['track_id'])} | {row['stability_flag']} | "
                f"{int(row['detections'])} | {float(row['coverage_rate']):.2f} | "
                f"{int(row['big_jump_count'])} | {int(row['nearest_rows'])} | "
                f"{int(row['controlled_nearest_rows'])} | {int(row['possession_rows'])} |"
            )

    report_path = output_dir / "track_stability_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simple persistent player-track stability."
    )
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--association", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="track_eval")
    parser.add_argument("--min-stable-seconds", type=float, default=2.0)
    parser.add_argument("--jump-px-per-frame", type=float, default=35.0)
    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks)
    association = pd.read_csv(args.association) if args.association else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_summary, fps, min_stable_frames = summarize_tracks(
        tracks,
        association=association,
        min_stable_seconds=args.min_stable_seconds,
        jump_px_per_frame=args.jump_px_per_frame,
    )
    summary_path = output_dir / "track_stability_summary.csv"
    track_summary.to_csv(summary_path, index=False)

    trouble_path = output_dir / "track_stability_trouble_tracks.csv"
    trouble = track_summary[
        (track_summary["stability_flag"] != "stable")
        & (
            (track_summary["nearest_rows"] > 0)
            | (track_summary["controlled_nearest_rows"] > 0)
            | (track_summary["possession_rows"] > 0)
        )
    ].sort_values(
        ["possession_rows", "controlled_nearest_rows", "nearest_rows", "detections"],
        ascending=False,
    )
    trouble.to_csv(trouble_path, index=False)

    report_path = build_report(
        track_summary,
        fps=fps,
        min_stable_frames=min_stable_frames,
        output_dir=output_dir,
        label=args.label,
    )

    print("PLAYER TRACK STABILITY EVALUATION COMPLETE")
    print("------------------------------------------")
    print(f"Tracks: {args.tracks}")
    if args.association:
        print(f"Association: {args.association}")
    print(f"Track summary: {summary_path}")
    print(f"Trouble tracks: {trouble_path}")
    print(f"Report: {report_path}")
    print("")
    print("Flags:")
    print(track_summary["stability_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
