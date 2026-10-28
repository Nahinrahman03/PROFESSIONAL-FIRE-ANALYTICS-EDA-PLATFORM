from __future__ import annotations
import math
import numpy as np
import pandas as pd


def json_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{key: json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]
