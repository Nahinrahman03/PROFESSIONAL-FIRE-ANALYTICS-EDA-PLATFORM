"""Feature engineering for FireScope ML models.

Builds a clean numeric feature matrix from the detected dataset columns,
handling missing values, encoding, and scaling consistently.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Semantics that make useful predictive features
FEATURE_SEMANTICS = ["latitude", "longitude", "brightness", "frp", "confidence"]
CATEGORICAL_SEMANTICS = ["source", "daynight"]


def build_regression_features(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Build (X, y, feature_names) for FRP regression.

    Target: frp column.
    Returns None if the target column is absent or has too few valid rows.
    """
    target_col = detected.get("frp")
    if not target_col or frame[target_col].dropna().shape[0] < 10:
        return None

    feature_cols, names = _numeric_features(frame, detected, exclude={"frp"})
    if not feature_cols:
        return None

    X_raw = frame[feature_cols].values.astype(float)
    y_raw = frame[target_col].values.astype(float)

    # Keep only rows where target is valid
    valid_mask = np.isfinite(y_raw)
    X_raw, y_raw = X_raw[valid_mask], y_raw[valid_mask]
    if len(y_raw) < 10:
        return None

    # Downsample to prevent MemoryError on massive datasets
    if len(y_raw) > 100000:
        idx = np.random.default_rng(42).choice(len(y_raw), 100000, replace=False)
        X_raw, y_raw = X_raw[idx], y_raw[idx]

    X = _impute_scale(X_raw)
    return X, y_raw, names


def build_classification_features(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
    """Build (X, y_encoded, feature_names, class_names) for confidence classification.

    Target: confidence column binarised to High / Low.
    Returns None if target absent or too few rows.
    """
    conf_col = detected.get("confidence")
    if not conf_col or frame[conf_col].dropna().shape[0] < 20:
        return None

    feature_cols, names = _numeric_features(frame, detected, exclude={"confidence"})
    if not feature_cols:
        return None

    raw_conf = frame[conf_col].copy()
    # Convert to H / M / L label
    numeric = pd.to_numeric(raw_conf, errors="coerce")
    if numeric.notna().mean() > 0.6:
        labels = pd.cut(
            numeric,
            bins=[-np.inf, 50, 80, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    else:
        labels = raw_conf.astype(str).str.strip().str.title()
        labels = labels.replace({"H": "High", "M": "Medium", "L": "Low"})

    X_raw = frame[feature_cols].values.astype(float)
    valid_mask = labels.isin(["Low", "Medium", "High"]).values
    X_raw, y_labels = X_raw[valid_mask], labels[valid_mask].values

    if len(y_labels) < 20:
        return None

    # Downsample to prevent MemoryError on massive datasets
    if len(y_labels) > 100000:
        idx = np.random.default_rng(42).choice(len(y_labels), 100000, replace=False)
        X_raw, y_labels = X_raw[idx], y_labels[idx]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    X = _impute_scale(X_raw)
    return X, y_encoded, names, list(le.classes_)


def build_clustering_features(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
) -> tuple[np.ndarray, list[str], pd.DataFrame] | None:
    """Build (X, feature_names, df_sampled) for K-Means clustering.

    Uses lat/lon + frp + brightness where available.
    Returns None if fewer than 20 valid rows.
    """
    preferred = ["latitude", "longitude", "frp", "brightness"]
    cols, names = [], []
    for sem in preferred:
        col = detected.get(sem)
        if col and col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
            names.append(sem)

    if len(cols) < 2:
        return None

    df_valid = frame[cols].dropna()
    if len(df_valid) < 20:
        return None

    # Downsample to prevent MemoryError on massive datasets
    if len(df_valid) > 50000:
        df_valid = df_valid.sample(50000, random_state=42)

    X_raw = df_valid.values.astype(float)
    X = _impute_scale(X_raw)
    return X, names, df_valid


# ── helpers ────────────────────────────────────────────────────────────

def _numeric_features(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
    exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (column_names, semantic_labels) for available numeric feature semantics."""
    exclude = exclude or set()
    cols, names = [], []
    for sem in FEATURE_SEMANTICS:
        if sem in exclude:
            continue
        col = detected.get(sem)
        if col and col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
            names.append(sem)
    return cols, names


def _impute_scale(X: np.ndarray) -> np.ndarray:
    """Median-impute then standard-scale a 2-D float array."""
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    X = StandardScaler().fit_transform(X)
    return X

