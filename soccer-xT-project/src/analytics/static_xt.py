import numpy as np
import pandas as pd


def get_bin(value, max_value, n_bins):
    """
    Convert a coordinate value into a grid bin.
    """
    if pd.isna(value):
        return np.nan

    bin_id = int((value / max_value) * n_bins)

    return min(max(bin_id, 0), n_bins - 1)


def add_static_xt(
    actions,
    xt_grid,
    pitch_length=105,
    pitch_width=68
):
    """
    Assign xT values from a static/pretrained xT grid.
    
    actions must include:
    - start_x
    - start_y
    - end_x
    - end_y
    """

    actions = actions.copy()

    x_bins = xt_grid.shape[1]
    y_bins = xt_grid.shape[0]

    actions["start_x_bin"] = actions["start_x"].apply(
        lambda x: get_bin(x, pitch_length, x_bins)
    )

    actions["start_y_bin"] = actions["start_y"].apply(
        lambda y: get_bin(y, pitch_width, y_bins)
    )

    actions["end_x_bin"] = actions["end_x"].apply(
        lambda x: get_bin(x, pitch_length, x_bins)
    )

    actions["end_y_bin"] = actions["end_y"].apply(
        lambda y: get_bin(y, pitch_width, y_bins)
    )

    actions["start_xT"] = actions.apply(
        lambda row: xt_grid[
            int(row["start_y_bin"]),
            int(row["start_x_bin"])
        ],
        axis=1
    )

    actions["end_xT"] = actions.apply(
        lambda row: xt_grid[
            int(row["end_y_bin"]),
            int(row["end_x_bin"])
        ],
        axis=1
    )

    actions["xT_added"] = actions["end_xT"] - actions["start_xT"]

    return actions


def player_xt_table(actions_with_xt, player_col="player_name"):
    """
    Aggregate xT by player.
    """

    return (
        actions_with_xt
        .groupby(player_col, as_index=False)
        .agg(
            total_xT=("xT_added", "sum"),
            actions=("xT_added", "count"),
            avg_xT=("xT_added", "mean")
        )
        .sort_values("total_xT", ascending=False)
    )