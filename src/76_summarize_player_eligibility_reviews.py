#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


NON_ACTIVE_LABELS = {"referee", "substitute_or_staff", "off_field_player"}
KNOWN_LABELS = NON_ACTIVE_LABELS | {"active_player", "unclear"}


def clean_id(value):
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def summarize_player_eligibility_reviews(review_csv, output_dir):
    review_csv = Path(review_csv)
    output_dir = Path(output_dir)

    reviews = pd.read_csv(review_csv)
    required = {
        "review_id",
        "frame",
        "time_seconds",
        "nearest_player_id",
        "eligibility_label",
    }
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"Missing required review columns: {missing}")

    reviews = reviews.copy()
    reviews["eligibility_label"] = reviews["eligibility_label"].fillna("").str.strip()
    reviews["nearest_player_id_clean"] = reviews["nearest_player_id"].map(clean_id)
    reviews["is_completed"] = reviews["eligibility_label"] != ""
    reviews["is_known_label"] = reviews["eligibility_label"].isin(KNOWN_LABELS)
    reviews["truth_binary"] = reviews["eligibility_label"].map(
        lambda label: "non_active"
        if label in NON_ACTIVE_LABELS
        else "active_player"
        if label == "active_player"
        else "unclear"
        if label == "unclear"
        else ""
    )

    completed = reviews[reviews["is_completed"]].copy()
    if completed.empty:
        summary = pd.DataFrame(
            columns=[
                "nearest_player_id",
                "reviewed_rows",
                "known_label_rows",
                "active_player_votes",
                "non_active_votes",
                "unclear_votes",
                "majority_label",
                "majority_binary",
                "majority_share",
                "first_frame",
                "last_frame",
                "first_time_seconds",
                "last_time_seconds",
                "labels_seen",
            ]
        )
    else:
        rows = []
        for player_id, group in completed.groupby("nearest_player_id_clean", dropna=False):
            label_counts = group["eligibility_label"].value_counts(dropna=False)
            binary_counts = group["truth_binary"].value_counts(dropna=False)
            known = group[group["is_known_label"]]
            majority_label = label_counts.index[0] if len(label_counts) else ""
            majority_binary = binary_counts.index[0] if len(binary_counts) else ""
            majority_share = (
                float(label_counts.iloc[0] / len(group)) if len(label_counts) else 0.0
            )
            rows.append(
                {
                    "nearest_player_id": player_id,
                    "reviewed_rows": int(len(group)),
                    "known_label_rows": int(len(known)),
                    "active_player_votes": int(
                        (group["eligibility_label"] == "active_player").sum()
                    ),
                    "non_active_votes": int(
                        group["eligibility_label"].isin(NON_ACTIVE_LABELS).sum()
                    ),
                    "unclear_votes": int(
                        (group["eligibility_label"] == "unclear").sum()
                    ),
                    "majority_label": majority_label,
                    "majority_binary": majority_binary,
                    "majority_share": majority_share,
                    "first_frame": int(group["frame"].min()),
                    "last_frame": int(group["frame"].max()),
                    "first_time_seconds": float(group["time_seconds"].min()),
                    "last_time_seconds": float(group["time_seconds"].max()),
                    "labels_seen": ";".join(label_counts.index.astype(str)),
                }
            )
        summary = pd.DataFrame(rows).sort_values(
            ["non_active_votes", "active_player_votes", "reviewed_rows"],
            ascending=[False, False, False],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed_player_eligibility_reviews.csv"
    summary_path = output_dir / "player_eligibility_track_summary.csv"
    rows_path = output_dir / "player_eligibility_review_rows_with_status.csv"
    reviews.to_csv(rows_path, index=False)
    completed.to_csv(completed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("PLAYER ELIGIBILITY REVIEW SUMMARY COMPLETE")
    print("-----------------------------------------")
    print(f"Review rows: {len(reviews)}")
    print(f"Completed rows: {len(completed)}")
    print(f"Uncompleted rows: {len(reviews) - len(completed)}")
    print(f"Reviewed player IDs: {len(summary)}")
    print(f"Rows with status: {rows_path}")
    print(f"Completed rows: {completed_path}")
    print(f"Track summary: {summary_path}")
    if len(completed):
        print("")
        print("Labels:")
        print(completed["eligibility_label"].value_counts(dropna=False).to_string())
        print("")
        print("Binary labels:")
        print(completed["truth_binary"].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed player eligibility reviews by nearest player/track ID."
        )
    )
    parser.add_argument("review_csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summarize_player_eligibility_reviews(
        review_csv=args.review_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
