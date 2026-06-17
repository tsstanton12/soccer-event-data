#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


SUCCESS_OUTCOMES = {"true_positive", "true_negative_null"}
VISIBLE_OUTCOMES = {"true_positive", "poor_localization", "missed_ball"}


def metrics(df):
    visible = df[df["gt_box_count"] > 0]
    nulls = df[df["gt_box_count"] == 0]
    return {
        "rows": len(df),
        "visible_rows": len(visible),
        "null_rows": len(nulls),
        "true_positive": int((df["outcome"] == "true_positive").sum()),
        "missed_ball": int((df["outcome"] == "missed_ball").sum()),
        "poor_localization": int((df["outcome"] == "poor_localization").sum()),
        "false_positive_on_null": int((df["outcome"] == "false_positive_on_null").sum()),
        "true_negative_null": int((df["outcome"] == "true_negative_null").sum()),
        "visible_recall": float((visible["outcome"] == "true_positive").mean()) if len(visible) else None,
        "null_false_positive_rate": float((nulls["outcome"] == "false_positive_on_null").mean()) if len(nulls) else None,
        "poor_localization_rate": float((visible["outcome"] == "poor_localization").mean()) if len(visible) else None,
        "mean_best_iou_visible": float(visible["best_iou"].mean()) if len(visible) else None,
    }


def outcome_score(outcome):
    if outcome == "true_positive":
        return 3
    if outcome == "true_negative_null":
        return 2
    if outcome == "poor_localization":
        return 1
    return 0


def image_key(df):
    if "clean_image_path" in df.columns:
        return df["clean_image_path"].map(lambda value: Path(str(value)).stem)
    if "image_path" in df.columns:
        return df["image_path"].map(lambda value: Path(str(value)).stem)
    if {"venue", "frame"}.issubset(df.columns):
        return df["venue"].astype(str) + ":" + df["frame"].astype(int).astype(str)
    raise ValueError("Evaluation details need clean_image_path/image_path or venue+frame.")


def compare(old_csv, new_csv, output_dir, old_name="old_model", new_name="new_model"):
    old = pd.read_csv(old_csv).copy()
    new = pd.read_csv(new_csv).copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old["_key"] = image_key(old)
    new["_key"] = image_key(new)

    merged = old.merge(
        new,
        on="_key",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    merged["old_score"] = merged["outcome_old"].fillna("").map(outcome_score)
    merged["new_score"] = merged["outcome_new"].fillna("").map(outcome_score)
    merged["delta_score"] = merged["new_score"] - merged["old_score"]
    merged["comparison_result"] = merged["delta_score"].map(
        lambda delta: "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    )

    details_path = output_dir / "ball_model_evaluation_comparison_details.csv"
    merged.to_csv(details_path, index=False)

    metric_rows = []
    for name, df in [(old_name, old), (new_name, new)]:
        metric_rows.append({"model": name, **metrics(df)})
    metric_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "ball_model_evaluation_comparison_metrics.csv"
    metric_df.to_csv(metrics_path, index=False)

    group_rows = []
    for group_cols in [["venue"], ["reason"], ["venue", "reason"]]:
        old_cols = [f"{col}_old" for col in group_cols]
        if not set(old_cols).issubset(merged.columns):
            continue
        grouped = (
            merged.groupby(old_cols)
            .agg(
                rows=("_key", "count"),
                improved=("comparison_result", lambda s: int((s == "improved").sum())),
                regressed=("comparison_result", lambda s: int((s == "regressed").sum())),
                unchanged=("comparison_result", lambda s: int((s == "unchanged").sum())),
                mean_delta_score=("delta_score", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "grouping", "+".join(group_cols))
        grouped = grouped.rename(columns={f"{col}_old": col for col in group_cols})
        group_rows.append(grouped)
    grouped_df = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    grouped_path = output_dir / "ball_model_evaluation_comparison_by_group.csv"
    grouped_df.to_csv(grouped_path, index=False)

    improved = int((merged["comparison_result"] == "improved").sum())
    regressed = int((merged["comparison_result"] == "regressed").sum())
    unchanged = int((merged["comparison_result"] == "unchanged").sum())

    report_lines = [
        "# Ball Model Evaluation Comparison",
        "",
        f"Old model: `{old_name}` / `{old_csv}`",
        f"New model: `{new_name}` / `{new_csv}`",
        "",
        f"Compared rows: {len(merged)}",
        f"Improved rows: {improved}",
        f"Regressed rows: {regressed}",
        f"Unchanged rows: {unchanged}",
        "",
        "Metrics:",
        "",
        metric_df.to_string(index=False),
        "",
        "Outputs:",
        "",
        f"- `{details_path}`",
        f"- `{metrics_path}`",
        f"- `{grouped_path}`",
    ]
    report_path = output_dir / "ball_model_evaluation_comparison_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print("BALL MODEL EVALUATION COMPARISON COMPLETE")
    print("-----------------------------------------")
    print(f"Compared rows: {len(merged)}")
    print(f"Improved: {improved}")
    print(f"Regressed: {regressed}")
    print(f"Unchanged: {unchanged}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two hard-frame ball model evaluation detail CSVs."
    )
    parser.add_argument("--old-evaluation", required=True)
    parser.add_argument("--new-evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-name", default="old_model")
    parser.add_argument("--new-name", default="new_model")
    args = parser.parse_args()

    compare(
        old_csv=args.old_evaluation,
        new_csv=args.new_evaluation,
        output_dir=args.output_dir,
        old_name=args.old_name,
        new_name=args.new_name,
    )


if __name__ == "__main__":
    main()
