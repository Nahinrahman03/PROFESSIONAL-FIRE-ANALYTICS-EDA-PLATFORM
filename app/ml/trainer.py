"""ML trainer functions for FireScope.

Three independent training workflows:
  - train_regression   → Random Forest regression on FRP
  - train_clustering   → K-Means geographic/intensity clustering
  - train_classification → Random Forest classification on confidence labels

Each returns a plain dict suitable for JSON serialisation.
Models are saved to data/models/ via joblib.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    silhouette_score,
)
from app.config.settings import PROJECT_ROOT

from app.ml.feature_builder import (
    build_regression_features,
    build_classification_features,
    build_clustering_features,
)


def _models_dir() -> Path:
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            root = Path(current_app.root_path).parent
        else:
            root = PROJECT_ROOT
    except Exception:
        root = PROJECT_ROOT
    d = root / "data" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_model(model, name: str) -> str:
    """Persist model to disk and return its filename."""
    filename = f"{name}.pkl"
    joblib.dump(model, _models_dir() / filename)
    return filename


# ─── Regression ───────────────────────────────────────────────────────

def train_regression(
    frame,
    detected: dict[str, str | None],
    n_estimators: int = 100,
) -> dict:
    result = build_regression_features(frame, detected)
    if result is None:
        return {
            "available": False,
            "message": "FRP column not found or too few valid rows to train a regression model.",
        }

    X, y, feature_names = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    t0 = time.monotonic()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = round(time.monotonic() - t0, 2)

    y_pred = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
    r2 = round(float(r2_score(y_test, y_pred)), 4)

    importances = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: -x[1],
        )
    ]

    # Scatter: actual vs predicted (sample up to 500 points for chart)
    sample_n = min(500, len(y_test))
    idx = np.random.default_rng(0).choice(len(y_test), sample_n, replace=False)
    scatter = [
        {"actual": round(float(y_test[i]), 2), "predicted": round(float(y_pred[i]), 2)}
        for i in idx
    ]

    filename = _save_model(model, "regression_frp")

    return {
        "available": True,
        "model": "regression",
        "label": "Fire Intensity Predictor",
        "algorithm": "Random Forest Regressor",
        "training_seconds": elapsed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "feature_importances": importances,
        "scatter": scatter,
        "download_name": filename,
    }


# ─── Clustering ───────────────────────────────────────────────────────

def train_clustering(
    frame,
    detected: dict[str, str | None],
    n_clusters: int = 5,
) -> dict:
    result = build_clustering_features(frame, detected)
    if result is None:
        return {
            "available": False,
            "message": "Insufficient geographic/numeric columns for clustering (need at least lat + lon).",
        }

    X, feature_names, df_sampled = result
    n_clusters = max(2, min(n_clusters, min(15, len(X) - 1)))

    t0 = time.monotonic()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    elapsed = round(time.monotonic() - t0, 2)

    sil = round(float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)), 4)
    inertia = round(float(model.inertia_), 2)

    counts = {str(i): int((labels == i).sum()) for i in range(n_clusters)}
    cluster_sizes = [{"cluster": f"Cluster {i+1}", "count": int((labels == i).sum())} for i in range(n_clusters)]

    # Centroids in original space for display using matched sample
    lat_col = detected.get("latitude")
    lon_col = detected.get("longitude")
    centroids = []
    if lat_col and lon_col and lat_col in df_sampled.columns and lon_col in df_sampled.columns:
        df_clustered = df_sampled[[lat_col, lon_col]].copy()
        df_clustered["_cluster"] = labels
        for i in range(n_clusters):
            grp = df_clustered[df_clustered["_cluster"] == i]
            if not grp.empty:
                centroids.append({
                    "cluster": i + 1,
                    "lat": round(float(grp[lat_col].mean()), 4),
                    "lon": round(float(grp[lon_col].mean()), 4),
                    "count": int(len(grp)),
                })

    filename = _save_model(model, "clustering_kmeans")

    return {
        "available": True,
        "model": "clustering",
        "label": "Hotspot Clustering",
        "algorithm": f"K-Means (k={n_clusters})",
        "training_seconds": elapsed,
        "n_clusters": n_clusters,
        "features_used": feature_names,
        "metrics": {"silhouette": sil, "inertia": inertia, "n_points": len(X)},
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
        "download_name": filename,
    }


# ─── Classification ───────────────────────────────────────────────────

def train_classification(
    frame,
    detected: dict[str, str | None],
    n_estimators: int = 100,
) -> dict:
    result = build_classification_features(frame, detected)
    if result is None:
        return {
            "available": False,
            "message": "Confidence column not found, cannot be encoded, or too few valid rows.",
        }

    X, y, feature_names, class_names = result
    counts = np.bincount(y)
    can_stratify = len(counts) > 1 and np.min(counts) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )


    t0 = time.monotonic()
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    elapsed = round(time.monotonic() - t0, 2)

    y_pred = model.predict(X_test)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    per_class = [
        {
            "label": cls,
            "precision": round(float(report[cls]["precision"]), 3),
            "recall": round(float(report[cls]["recall"]), 3),
            "f1": round(float(report[cls]["f1-score"]), 3),
            "support": int(report[cls]["support"]),
        }
        for cls in class_names if cls in report
    ]

    importances = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: -x[1],
        )
    ]

    filename = _save_model(model, "classification_confidence")

    return {
        "available": True,
        "model": "classification",
        "label": "Confidence Classifier",
        "algorithm": "Random Forest Classifier",
        "training_seconds": elapsed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_names": class_names,
        "metrics": {"accuracy": accuracy},
        "per_class": per_class,
        "feature_importances": importances,
        "download_name": filename,
    }
