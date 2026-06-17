#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def command_text(command):
    return " ".join(str(part) for part in command)


def run_step(title, command, dry_run=False):
    print("")
    print(title)
    print("-" * len(title))
    print(command_text(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def validate_path(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def build_pipeline(args):
    video = Path(args.video)
    players = Path(args.players)
    track_stability = Path(args.track_stability)
    model = Path(args.model) if args.model else None
    detections_csv = Path(args.detections_csv) if args.detections_csv else None

    run_name = args.run_name or f"{args.venue_label}_new_ball_model"
    output_dir = Path("outputs") / run_name

    if detections_csv is None:
        detections_csv = output_dir / f"{video.stem}_ball_detections.csv"

    interpolated_csv = output_dir / f"{video.stem}_extended_interpolated_ball_path.csv"
    association_csv = output_dir / f"{args.venue_label}_ball_player_association_tracked.csv"
    min_label = f"min{args.min_confirm_frames}"
    chain_frames_csv = output_dir / f"{args.venue_label}_{min_label}_stability_gated_frames.csv"
    chain_segments_csv = output_dir / f"{args.venue_label}_{min_label}_stability_gated_segments.csv"
    overlay_path = output_dir / f"{args.venue_label}_{min_label}_stability_gated_overlay_first{int(args.overlay_seconds)}s.mp4"
    summary_path = output_dir / "ball_model_rerun_summary.md"

    commands = []
    if args.detections_csv is None:
        if model is None:
            raise ValueError("Provide --model or --detections-csv.")
        commands.append(
            (
                "1. Detect ball candidates",
                [
                    sys.executable,
                    "src/08_detect_ball_custom_model.py",
                    "--venue",
                    run_name,
                    "--video",
                    str(video),
                    "--model",
                    str(model),
                    "--conf",
                    str(args.conf),
                ],
            )
        )
    else:
        print(f"Using existing detections CSV: {detections_csv}")

    commands.extend(
        [
            (
                "2. Cautiously interpolate ball path",
                [
                    sys.executable,
                    "src/38_interpolate_ball_path_extended.py",
                    str(detections_csv),
                    "--output",
                    str(interpolated_csv),
                    "--fps",
                    str(args.fps),
                    "--max-gap-seconds",
                    str(args.max_gap_seconds),
                    "--max-speed",
                    str(args.max_speed),
                ],
            ),
            (
                "3. Associate ball to tracked players",
                [
                    sys.executable,
                    "src/41_ball_player_association_v2.py",
                    str(interpolated_csv),
                    str(players),
                    "--output",
                    str(association_csv),
                    "--fps",
                    str(args.fps),
                    "--max-distance-px",
                    str(args.max_distance_px),
                    "--control-distance-px",
                    str(args.control_distance_px),
                    "--controlled-speed",
                    str(args.controlled_speed),
                    "--transit-speed",
                    str(args.transit_speed),
                    "--control-field-zones",
                    args.control_field_zones,
                    "--recompute-time-seconds",
                ],
            ),
            (
                "4. Build stability-gated possession chains",
                [
                    sys.executable,
                    "src/52_build_possession_chains.py",
                    str(association_csv),
                    "--output-frames",
                    str(chain_frames_csv),
                    "--output-segments",
                    str(chain_segments_csv),
                    "--fps",
                    str(args.fps),
                    "--min-confirm-frames",
                    str(args.min_confirm_frames),
                    "--max-control-distance-px",
                    str(args.max_control_distance_px),
                    "--max-carry-forward-seconds",
                    str(args.max_carry_forward_seconds),
                    "--track-stability-csv",
                    str(track_stability),
                    "--exclude-track-stability-flags",
                    args.exclude_track_stability_flags,
                ],
            ),
        ]
    )

    if args.exclude_active_participant_decisions:
        commands[-1][1].extend(
            [
                "--exclude-active-participant-decisions",
                args.exclude_active_participant_decisions,
            ]
        )

    if not args.no_overlay:
        commands.append(
            (
                "5. Render first-window possession overlay",
                [
                    sys.executable,
                    "src/53_visualize_possession_chains.py",
                    str(video),
                    str(chain_frames_csv),
                    "--output",
                    str(overlay_path),
                    "--max-frames",
                    str(int(round(args.overlay_seconds * args.fps))),
                ],
            )
        )

    outputs = {
        "output_dir": output_dir,
        "detections_csv": detections_csv,
        "interpolated_csv": interpolated_csv,
        "association_csv": association_csv,
        "chain_frames_csv": chain_frames_csv,
        "chain_segments_csv": chain_segments_csv,
        "overlay_path": overlay_path if not args.no_overlay else None,
        "summary_path": summary_path,
    }
    return commands, outputs


def write_summary(args, outputs):
    summary_path = outputs["summary_path"]
    lines = [
        "# Ball Model Rerun Summary",
        "",
        f"- Venue label: `{args.venue_label}`",
        f"- Video: `{args.video}`",
        f"- Players: `{args.players}`",
        f"- Track stability: `{args.track_stability}`",
        f"- FPS: `{args.fps}`",
        f"- Min confirm frames: `{args.min_confirm_frames}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key == "summary_path" or value is None:
            continue
        lines.append(f"- {key}: `{value}`")
    summary_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a new ball model through full-clip detection, interpolation, "
            "tracked-player association, stability-gated possession chains, "
            "and an optional first-window overlay."
        )
    )
    parser.add_argument("--venue-label", required=True, help="Short label like lemoyne or army.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--players", required=True, help="Tracked player CSV.")
    parser.add_argument("--track-stability", required=True, help="Track stability summary CSV.")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--model", default=None, help="YOLO/Roboflow model path.")
    parser.add_argument(
        "--detections-csv",
        default=None,
        help="Use an existing ball detection CSV and skip model inference.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output namespace under outputs/. Default: <venue-label>_new_ball_model.",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--max-gap-seconds", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=2500)
    parser.add_argument("--max-distance-px", type=float, default=150)
    parser.add_argument("--control-distance-px", type=float, default=80)
    parser.add_argument("--controlled-speed", type=float, default=500)
    parser.add_argument("--transit-speed", type=float, default=700)
    parser.add_argument("--control-field-zones", default="strict,tolerant")
    parser.add_argument("--min-confirm-frames", type=int, default=15)
    parser.add_argument("--max-control-distance-px", type=float, default=80)
    parser.add_argument("--max-carry-forward-seconds", type=float, default=2.0)
    parser.add_argument(
        "--exclude-track-stability-flags",
        default="short_track,jumpy_track,gappy_track",
    )
    parser.add_argument(
        "--exclude-active-participant-decisions",
        default="",
        help=(
            "Optional active-participant advisory decisions to exclude from "
            "control, if the association CSV contains those columns."
        ),
    )
    parser.add_argument("--overlay-seconds", type=float, default=60)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without running them.",
    )
    args = parser.parse_args()

    validate_path(args.video, "Video")
    validate_path(args.players, "Tracked player CSV")
    validate_path(args.track_stability, "Track stability CSV")
    if args.detections_csv:
        validate_path(args.detections_csv, "Detection CSV")
    elif args.model and not args.dry_run:
        validate_path(args.model, "Model")

    commands, outputs = build_pipeline(args)
    if not args.dry_run:
        outputs["output_dir"].mkdir(parents=True, exist_ok=True)

    print("BALL MODEL FULL-CLIP RERUN")
    print("--------------------------")
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Dry run: {args.dry_run}")

    for title, command in commands:
        run_step(title, command, dry_run=args.dry_run)

    if not args.dry_run:
        write_summary(args, outputs)
        print("")
        print("RERUN COMPLETE")
        print("--------------")
        print(f"Summary: {outputs['summary_path']}")
        if outputs["overlay_path"] is not None:
            print(f"Overlay: {outputs['overlay_path']}")


if __name__ == "__main__":
    main()
