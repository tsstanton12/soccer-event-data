#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def normalize_yes_no(value):
    return str(value or "").strip().lower()


def desired_states_from_note(row):
    note = str(row.get("review_notes") or "").lower().strip()
    is_correct = normalize_yes_no(row.get("review_correct")) == "yes"

    if is_correct:
        return row["before_state"], row["switch_state"]

    if (
        "controlled in first frame" in note
        and "in_transit in second frame" in note
    ):
        return "controlled", "in_transit"
    if (
        "loose in first frame" in note
        and "controlled in second frame" in note
    ):
        return "loose_or_unclear", "controlled"
    if "go from controlled to in_transit" in note or "go from controlled to transit" in note:
        return "controlled", "in_transit"
    if "go from in_transit to controlled" in note or "go from transit to controlled" in note:
        return "in_transit", "controlled"
    if (
        "stay controlled" in note
        or "marked controlled" in note
        or "marked as controlled" in note
        or "controlled in both" in note
    ):
        return "controlled", "controlled"
    if (
        "in_transit in both" in note
        or "marked in_transit in both" in note
        or "remain classed as in_transit" in note
        or "this is a pass" in note
        or "long pass" in note
    ):
        return "in_transit", "in_transit"
    if "marked loose" in note or "loose in both" in note:
        return "loose_or_unclear", "loose_or_unclear"

    return None, None


def build_frame_labels(completed_manifest):
    manifest = pd.read_csv(completed_manifest)
    rows = []

    for _, row in manifest.iterrows():
        desired_before, desired_switch = desired_states_from_note(row)
        if desired_before is not None:
            rows.append({
                "review_id": row["review_id"],
                "side": "before",
                "frame": int(row["before_frame"]),
                "time_seconds": float(row["before_time"]),
                "model_state": row["before_state"],
                "desired_state": desired_before,
                "review_correct": row.get("review_correct"),
                "review_notes": row.get("review_notes"),
            })
        if desired_switch is not None:
            rows.append({
                "review_id": row["review_id"],
                "side": "switch",
                "frame": int(row["switch_frame"]),
                "time_seconds": float(row["switch_time"]),
                "model_state": row["switch_state"],
                "desired_state": desired_switch,
                "review_correct": row.get("review_correct"),
                "review_notes": row.get("review_notes"),
            })

    return pd.DataFrame(rows)


def evaluate_predictions(frame_labels, association_csv, state_column="ball_state"):
    association = pd.read_csv(association_csv)
    predictions = association[["frame", state_column]].rename(
        columns={state_column: "predicted_state"}
    )

    scored = frame_labels.merge(predictions, on="frame", how="left")
    scored["is_correct"] = scored["desired_state"] == scored["predicted_state"]
    return scored


def print_summary(scored):
    correct = int(scored["is_correct"].sum())
    total = len(scored)
    accuracy = correct / total if total else 0

    print("BALL STATE REVIEW EVALUATION")
    print("----------------------------")
    print(f"Reviewed frame labels: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print("")
    print("Desired vs predicted:")
    print(pd.crosstab(scored["desired_state"], scored["predicted_state"]).to_string())
    print("")
    print("Errors by desired/predicted state:")
    errors = scored[scored["is_correct"] == False]
    if len(errors):
        print(
            errors.groupby(["desired_state", "predicted_state"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
    else:
        print("No errors.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ball-state predictions against a completed transition-review manifest."
    )
    parser.add_argument("completed_manifest")
    parser.add_argument("association_csv")
    parser.add_argument("--state-column", default="ball_state")
    parser.add_argument("--output-labels")
    parser.add_argument("--output-scored")
    args = parser.parse_args()

    completed_manifest = Path(args.completed_manifest)
    association_csv = Path(args.association_csv)

    labels = build_frame_labels(completed_manifest)
    scored = evaluate_predictions(labels, association_csv, args.state_column)
    print_summary(scored)

    if args.output_labels:
        output_labels = Path(args.output_labels)
        output_labels.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(output_labels, index=False)
        print(f"Frame labels: {output_labels}")

    if args.output_scored:
        output_scored = Path(args.output_scored)
        output_scored.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_scored, index=False)
        print(f"Scored frames: {output_scored}")


if __name__ == "__main__":
    main()
