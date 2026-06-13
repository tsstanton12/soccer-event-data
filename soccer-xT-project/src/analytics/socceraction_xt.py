import pandas as pd
import socceraction.xthreat as xthreat


def fit_socceraction_xt(spadl_actions):
    """
    Fit socceraction's Expected Threat model on SPADL actions.
    """
    model = xthreat.ExpectedThreat(l=16, w=12)
    model.fit(spadl_actions)
    return model


def add_socceraction_xt(spadl_actions, model):
    """
    Add xT values to SPADL actions using a fitted socceraction xT model.
    """
    actions = spadl_actions.copy()

    actions["xT_added"] = model.rate(actions)

    return actions


def player_xt_table(actions_with_xt):
    """
    Aggregate socceraction xT values by player.
    """
    return (
        actions_with_xt
        .groupby(["player_id"], as_index=False)
        .agg(
            total_xT=("xT_added", "sum"),
            actions=("xT_added", "count"),
            avg_xT_per_action=("xT_added", "mean")
        )
        .sort_values("total_xT", ascending=False)
    )