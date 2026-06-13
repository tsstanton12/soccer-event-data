import pandas as pd


XT_ACTIONS = [
    "pass",
    "carry"
]


def extract_xt_actions(events):
    """
    Extract actions usable for xT models.
    """

    xt_actions = events[
        events["event_type"].isin(XT_ACTIONS)
    ].copy()

    xt_actions = xt_actions.dropna(
        subset=[
            "start_x",
            "start_y",
            "end_x",
            "end_y"
        ]
    )

    return xt_actions