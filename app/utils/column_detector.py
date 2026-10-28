"""Heuristics for detecting semantic columns without hardcoding a CSV schema."""
from __future__ import annotations
from typing import Iterable
import pandas as pd

ALIASES: dict[str, tuple[str, ...]] = {
    "latitude": ("latitude", "lat", "y"), "longitude": ("longitude", "lon", "lng", "long", "x"),
    "date": ("acq_date", "date", "acquisition_date", "datetime", "timestamp"),
    "time": ("acq_time", "time", "acquisition_time"), "frp": ("frp", "fire_radiative_power", "radiative_power"),
    "confidence": ("confidence", "conf", "certainty"), "source": ("satellite", "instrument", "source", "platform", "sensor"),
    "brightness": ("brightness", "bright_t31", "bright", "temperature"), "region": ("country", "region", "state", "province", "admin"),
    "daynight": ("daynight", "day_night", "day/night"),
}


def normalise(name: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in name).split())


def detect_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    normalised = {normalise(column): column for column in frame.columns}
    result: dict[str, str | None] = {}
    for semantic, aliases in ALIASES.items():
        result[semantic] = next((normalised[a] for a in aliases if a in normalised), None)
    return result


def column_kind(series: pd.Series, name: str, detected: dict[str, str | None]) -> str:
    if name in {detected.get("latitude"), detected.get("longitude")}:
        return "geographic"
    if name == detected.get("date") or pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return "categorical" if unique_ratio <= 0.5 else "text"
