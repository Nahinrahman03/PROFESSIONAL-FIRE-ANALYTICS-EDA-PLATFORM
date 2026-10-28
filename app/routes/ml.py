"""ML API routes — train, status, and model download endpoints."""
from __future__ import annotations
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file, current_app
from app.services.dataset_service import get_dataset, DatasetUnavailable
from app.ml.trainer import train_regression, train_clustering, train_classification

ml_bp = Blueprint("ml", __name__)


def _models_dir() -> Path:
    project_root = Path(current_app.root_path).parent
    return project_root / "data" / "models"


@ml_bp.errorhandler(DatasetUnavailable)
def dataset_error(error):
    return jsonify({"error": str(error)}), 503


@ml_bp.post("/train")
def train():
    """Train one or more models on the active dataset.

    Request body:
      {
        "models": ["regression", "clustering", "classification"],  // any subset
        "n_clusters": 5,       // optional, default 5
        "n_estimators": 100    // optional, default 100
      }
    """
    body = request.get_json(silent=True) or {}
    requested = set(body.get("models", ["regression", "clustering", "classification"]))
    n_clusters = int(body.get("n_clusters", 5))
    n_estimators = int(body.get("n_estimators", 100))

    dataset = get_dataset()
    frame = dataset.frame          # use full processed frame for training
    detected = dataset.columns

    results = {}
    if "regression" in requested:
        results["regression"] = train_regression(frame, detected, n_estimators)
    if "clustering" in requested:
        results["clustering"] = train_clustering(frame, detected, n_clusters)
    if "classification" in requested:
        results["classification"] = train_classification(frame, detected, n_estimators)

    if not results:
        return jsonify({"error": "No valid model type specified. Use 'regression', 'clustering', or 'classification'."}), 400

    return jsonify({"results": results})


@ml_bp.get("/download/<model_name>")
def download_model(model_name: str):
    """Stream a trained .pkl model file as a download."""
    # Whitelist allowed names to prevent path traversal
    allowed = {"regression_frp", "clustering_kmeans", "classification_confidence"}
    if model_name not in allowed:
        return jsonify({"error": "Unknown model name."}), 404

    model_path = _models_dir() / f"{model_name}.pkl"
    if not model_path.exists():
        return jsonify({"error": "Model has not been trained yet. Train it first."}), 404

    return send_file(
        model_path,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{model_name}.pkl",
    )
