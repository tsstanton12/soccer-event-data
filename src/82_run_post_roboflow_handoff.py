#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = "outputs/ball_detection_review/multi_venue_750/ball_detection_review_manifest.csv"
DEFAULT_IMAGE_DIR = "outputs/ball_detection_review/multi_venue_750/clean_frames"


FULL_CLIP_PRESETS = {
    "lemoyne": {
        "video": "videos/lemoyne_short_clip.mp4",
        "players": "outputs/player_tracking/lemoyne_players_on_field_tracked.csv",
        "track_stability": "outputs/player_tracking/track_stability/lemoyne_min15/track_stability_summary.csv",
        "fps": "30",
    },
    "army": {
        "video": "videos/army_short_clip.mp4",
        "players": "outputs/player_tracking/army_players_on_field_tracked.csv",
        "track_stability": "outputs/player_tracking/track_stability/army_min15/track_stability_summary.csv",
        "fps": "25",
    },
}


def command_text(command):
    return " ".join(str(part) for part in command)


def run_step(index, title, command, dry_run=False):
    print("")
    print(f"{index}. {title}")
    print("-" * (len(title) + len(str(index)) + 2))
    print(command_text(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def validate_path(path, label, allow_missing=False):
    if allow_missing:
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def parse_full_clip_list(value):
    if not value:
        return []
    lowered = value.strip().lower()
    if lowered == "none":
        return []
    if lowered == "all":
        return list(FULL_CLIP_PRESETS)
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def build_commands(args):
    output_root = Path(args.output_root)
    predictions_csv = output_root / "new_model_predictions.csv"
    validation_dir = output_root / "roboflow_export_validation"
    evaluation_dir = output_root / "new_model_evaluation"
    comparison_dir = output_root / "old_vs_new_model_comparison"

    commands = [
        (
            "Validate Roboflow YOLO export",
            [
                sys.executable,
                "src/80_validate_roboflow_ball_export.py",
                "--manifest",
                args.manifest,
                "--labels-dir",
                args.labels_dir,
                "--images-dir",
                args.image_dir,
                "--output-dir",
                str(validation_dir),
            ],
        ),
        (
            "Run new model on hard-frame images",
            [
                sys.executable,
                "src/75_detect_ball_review_images.py",
                "--model",
                args.model,
                "--image-dir",
                args.image_dir,
                "--output-csv",
                str(predictions_csv),
                "--conf",
                str(args.conf),
            ],
        ),
        (
            "Evaluate new model on hard-frame annotations",
            [
                sys.executable,
                "src/74_evaluate_ball_detection_review.py",
                "--manifest",
                args.manifest,
                "--labels-dir",
                args.labels_dir,
                "--predictions-csv",
                str(predictions_csv),
                "--output-dir",
                str(evaluation_dir),
                "--iou-threshold",
                str(args.iou_threshold),
                "--confidence-threshold",
                str(args.conf),
            ],
        ),
    ]

    if args.old_evaluation:
        commands.append(
            (
                "Compare old and new hard-frame evaluations",
                [
                    sys.executable,
                    "src/81_compare_ball_model_evaluations.py",
                    "--old-evaluation",
                    args.old_evaluation,
                    "--new-evaluation",
                    str(evaluation_dir / "ball_detection_review_evaluation_details.csv"),
                    "--output-dir",
                    str(comparison_dir),
                    "--old-name",
                    args.old_name,
                    "--new-name",
                    args.new_name,
                ],
            )
        )

    for venue in parse_full_clip_list(args.full_clips):
        if venue not in FULL_CLIP_PRESETS:
            raise ValueError(
                f"Unknown full-clip preset '{venue}'. Available: {', '.join(FULL_CLIP_PRESETS)}"
            )
        preset = FULL_CLIP_PRESETS[venue]
        commands.append(
            (
                f"Run full-clip rerun for {venue}",
                [
                    sys.executable,
                    "src/79_rerun_ball_model_pipeline.py",
                    "--venue-label",
                    venue,
                    "--video",
                    preset["video"],
                    "--players",
                    preset["players"],
                    "--track-stability",
                    preset["track_stability"],
                    "--fps",
                    preset["fps"],
                    "--model",
                    args.model,
                    "--run-name",
                    f"{venue}_{args.new_name}",
                    "--conf",
                    str(args.conf),
                ],
            )
        )

    outputs = {
        "output_root": output_root,
        "validation_report": validation_dir / "roboflow_ball_export_validation_report.md",
        "predictions_csv": predictions_csv,
        "evaluation_report": evaluation_dir / "ball_detection_review_evaluation_report.md",
        "evaluation_details": evaluation_dir / "ball_detection_review_evaluation_details.csv",
        "comparison_report": comparison_dir / "ball_model_evaluation_comparison_report.md"
        if args.old_evaluation
        else None,
    }
    return commands, outputs


def write_summary(args, outputs):
    summary_path = outputs["output_root"] / "post_roboflow_handoff_summary.md"
    lines = [
        "# Post-Roboflow Handoff Summary",
        "",
        f"- Model: `{args.model}`",
        f"- Labels dir: `{args.labels_dir}`",
        f"- Manifest: `{args.manifest}`",
        f"- Image dir: `{args.image_dir}`",
        f"- Confidence threshold: `{args.conf}`",
        f"- IoU threshold: `{args.iou_threshold}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key == "output_root" or value is None:
            continue
        lines.append(f"- {key}: `{value}`")
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the post-Roboflow handoff: validate labels, run the new model "
            "on hard frames, evaluate, optionally compare to an old evaluation, "
            "and optionally rerun full clips."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--output-root",
        default="outputs/ball_detection_review/post_roboflow_handoff",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--old-evaluation", default=None)
    parser.add_argument("--old-name", default="current_model")
    parser.add_argument("--new-name", default="roboflow_retrain")
    parser.add_argument(
        "--full-clips",
        default="none",
        help="Comma-separated presets to rerun: lemoyne,army,all,none.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_path(args.model, "Model", allow_missing=args.dry_run)
    validate_path(args.labels_dir, "Labels dir", allow_missing=args.dry_run)
    validate_path(args.manifest, "Manifest")
    validate_path(args.image_dir, "Image dir")
    if args.old_evaluation:
        validate_path(args.old_evaluation, "Old evaluation")

    commands, outputs = build_commands(args)
    if not args.dry_run:
        outputs["output_root"].mkdir(parents=True, exist_ok=True)

    print("POST-ROBOFLOW HANDOFF")
    print("---------------------")
    print(f"Output root: {outputs['output_root']}")
    print(f"Dry run: {args.dry_run}")

    for index, (title, command) in enumerate(commands, start=1):
        run_step(index, title, command, dry_run=args.dry_run)

    if not args.dry_run:
        summary_path = write_summary(args, outputs)
        print("")
        print("HANDOFF COMPLETE")
        print("----------------")
        print(f"Summary: {summary_path}")
        print(f"Evaluation report: {outputs['evaluation_report']}")
        if outputs["comparison_report"]:
            print(f"Comparison report: {outputs['comparison_report']}")


if __name__ == "__main__":
    main()
