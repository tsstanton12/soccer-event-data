import pandas as pd
import numpy as np


def add_simple_xt(
    actions,
    x_bins=12,
    y_bins=8
):
    """
    Adds simple grid-based xT values to pass/carry actions.

    This is a placeholder model for now.
    Later, we can replace this with socceraction's public xT model.
    """

    actions = actions.copy()

    # Create pitch bins
    actions["start_x_bin"] = pd.cut(
        actions["start_x"],
        bins=x_bins,
        labels=False,
        include_lowest=True
    )

    actions["start_y_bin"] = pd.cut(
        actions["start_y"],
        bins=y_bins,
        labels=False,
        include_lowest=True
    )

    actions["end_x_bin"] = pd.cut(
        actions["end_x"],
        bins=x_bins,
        labels=False,
        include_lowest=True
    )

    actions["end_y_bin"] = pd.cut(
        actions["end_y"],
        bins=y_bins,
        labels=False,
        include_lowest=True
    )

    # Temporary hand-built xT surface
    x_values = np.linspace(0.01, 0.35, x_bins)

    actions["start_xT"] = actions["start_x_bin"].apply(
        lambda x: x_values[int(x)] if pd.notna(x) else np.nan
    )

    actions["end_xT"] = actions["end_x_bin"].apply(
        lambda x: x_values[int(x)] if pd.notna(x) else np.nan
    )

    actions["xT_added"] = actions["end_xT"] - actions["start_xT"]

    return actions


def player_xt_table(actions_with_xt):
    """
    Creates a player xT leaderboard.
    """

    return (
        actions_with_xt
        .groupby(["player", "team"], as_index=False)
        .agg(
            total_xT=("xT_added", "sum"),
            actions=("xT_added", "count"),
            avg_xT_per_action=("xT_added", "mean")
        )
        .sort_values("total_xT", ascending=False)
    )