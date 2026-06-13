import pandas as pd
import numpy as np


STANDARD_COLUMNS = [
    "match_id",
    "period",
    "timestamp_seconds",
    "team",
    "player",
    "event_type",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "outcome",
    "possession_id",
    "source_event_id",
    "source"
]


def timestamp_to_seconds(timestamp):
    """
    Converts StatsBomb timestamp format to seconds.
    """

    if pd.isna(timestamp):
        return np.nan

    h, m, s = str(timestamp).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def convert_statsbomb_to_standard(events):
    """
    Convert StatsBomb events into our standard schema.
    """

    standard = pd.DataFrame()

    standard["match_id"] = events["match_id"]
    standard["period"] = events["period"]

    standard["timestamp_seconds"] = (
        events["timestamp"]
        .apply(timestamp_to_seconds)
    )

    standard["team"] = events["team_name"]
    standard["player"] = events["player_name"]

    standard["event_type"] = (
        events["type_name"]
        .str.lower()
    )

    standard["start_x"] = events["x"]
    standard["start_y"] = events["y"]

    standard["end_x"] = events["end_x"]
    standard["end_y"] = events["end_y"]

    standard["outcome"] = np.where(
        events["outcome_name"].isna(),
        "successful",
        events["outcome_name"].str.lower()
    )

    standard["possession_id"] = events["possession"]

    standard["source_event_id"] = events["id"]

    standard["source"] = "statsbomb"

    return standard[STANDARD_COLUMNS]