from __future__ import annotations
import os
import time
from io import StringIO
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
from werkzeug.utils import secure_filename
from app.services.dataset_service import get_dataset, DatasetUnavailable, list_available_datasets
from app.services.analytics_service import AnalyticsService
from app.services.alert_service import evaluate_rules
from app.utils.helpers import records

api_bp = Blueprint("api", __name__)


def payload() -> dict:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    args = request.args.to_dict(flat=False)
    if not args:
        return {}
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in args.items()}



def services() -> tuple:
    dataset = get_dataset()
    return dataset, AnalyticsService(dataset)


@api_bp.errorhandler(DatasetUnavailable)
def dataset_error(error):
    return jsonify({"error": str(error)}), 503


# ------------------------------------------------------------------
# Health / schema — rarely change; short cache headers
# ------------------------------------------------------------------

@api_bp.get("/health")
def health():
    dataset, _ = services()
    return jsonify({"status": "ok", "dataset": dataset.path.name, "records": len(dataset.raw)})


@api_bp.get("/schema")
def schema():
    dataset, _ = services()
    resp = jsonify(dataset.schema())
    resp.headers["Cache-Control"] = "private, max-age=60"
    return resp


@api_bp.route("/filter-options", methods=["GET", "POST"])
def filter_options():
    dataset, _ = services()
    frame = dataset.filter(payload())
    options = {}
    for semantic in ("source", "confidence", "daynight", "region"):
        col = dataset.columns.get(semantic)
        if col:
            options[semantic] = sorted(str(v) for v in frame[col].dropna().unique())[:100]
    for semantic in ("frp", "brightness"):
        col = dataset.columns.get(semantic)
        if col and frame[col].notna().any():
            options[semantic] = {"min": float(frame[col].min()), "max": float(frame[col].max())}
    date = dataset.columns.get("date")
    if date and frame[date].notna().any():
        options["dates"] = {"min": frame[date].min().date().isoformat(), "max": frame[date].max().date().isoformat()}
    return jsonify({"options": options, "detected": dataset.columns})


# ------------------------------------------------------------------
# Combined overview endpoint — replaces 7 parallel calls with 1
# ------------------------------------------------------------------

@api_bp.route("/overview", methods=["GET", "POST"])
def overview():
    """Return all overview payloads in one response to eliminate 6 round-trips."""
    dataset, analytics = services()
    params = payload()
    frame = dataset.filter(params)
    firing_alerts = evaluate_rules(dataset)
    return jsonify({
        "summary": analytics.summary(frame),
        "time_series": analytics.time_series(frame, str(params.get("period", "day"))),
        "frp": analytics.intensity(frame),
        "confidence": analytics.confidence_analysis(frame),
        "satellites": analytics.satellites(frame),
        "insights": {"items": analytics.insights(frame)},
        "map": analytics.map_data(frame),
        "alerts": firing_alerts,
    })


# ------------------------------------------------------------------
# Individual analytic endpoints (still used for non-overview views)
# ------------------------------------------------------------------

def filtered():
    dataset, analytics = services()
    return dataset.filter(payload()), analytics


@api_bp.route("/summary", methods=["GET", "POST"])
def summary():
    frame, analytics = filtered()
    return jsonify(analytics.summary(frame))


@api_bp.route("/time-series", methods=["GET", "POST"])
def time_series():
    frame, analytics = filtered()
    return jsonify(analytics.time_series(frame, str(payload().get("period", "day"))))


@api_bp.route("/frp", methods=["GET", "POST"])
def frp():
    frame, analytics = filtered()
    return jsonify(analytics.intensity(frame))


@api_bp.route("/confidence", methods=["GET", "POST"])
def confidence():
    frame, analytics = filtered()
    return jsonify(analytics.confidence_analysis(frame))


@api_bp.route("/satellites", methods=["GET", "POST"])
def satellites():
    frame, analytics = filtered()
    return jsonify(analytics.satellites(frame))


@api_bp.route("/geographic", methods=["GET", "POST"])
def geographic():
    frame, analytics = filtered()
    return jsonify(analytics.geography(frame))


@api_bp.route("/map", methods=["GET", "POST"])
def map_data():
    frame, analytics = filtered()
    return jsonify(analytics.map_data(frame))


@api_bp.route("/data-quality", methods=["GET", "POST"])
def data_quality():
    frame, analytics = filtered()
    return jsonify(analytics.data_quality(frame))


@api_bp.route("/insights", methods=["GET", "POST"])
def insights():
    frame, analytics = filtered()
    return jsonify({"items": analytics.insights(frame)})


@api_bp.route("/data", methods=["GET", "POST"])
def data_preview():
    frame, _ = filtered()
    params = payload()
    page = max(int(params.get("page", 1)), 1)
    page_size = min(max(int(params.get("page_size", 30)), 1), 100)
    start = (page - 1) * page_size
    return jsonify({
        "rows": records(frame.iloc[start:start + page_size]),
        "total": len(frame),
        "page": page,
        "page_size": page_size,
    })


@api_bp.route("/download", methods=["GET", "POST"])
def download():
    frame, _ = filtered()
    stream = StringIO()
    frame.to_csv(stream, index=False)
    return Response(
        stream.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fire-analytics-filtered.csv"},
    )


@api_bp.get("/datasets")
def get_datasets():
    """Return all available datasets with active indicator."""
    datasets = list_available_datasets()
    active_name = next((d["name"] for d in datasets if d["active"]), "")
    return jsonify({"datasets": datasets, "active": active_name})


@api_bp.post("/datasets/select")
def select_dataset():
    """Switch active dataset to the selected file."""
    body = payload()
    target_name = body.get("name") or body.get("filename")
    target_path = body.get("path")

    datasets = list_available_datasets()
    matched = None
    for d in datasets:
        if (target_path and d["path"] == target_path) or (target_name and d["name"] == target_name):
            matched = d
            break

    if not matched:
        return jsonify({"error": "Dataset not found among available files."}), 404

    current_app.config["DATASET_PATH"] = matched["path"]
    current_app.extensions.pop("dataset_service", None)

    try:
        dataset = get_dataset()
        return jsonify({
            "message": f"Switched active dataset to {matched['name']}.",
            "dataset": dataset.schema(),
            "active": matched["name"],
            "datasets": list_available_datasets(),
        })
    except DatasetUnavailable as err:
        return jsonify({"error": f"Failed to load dataset: {err}"}), 400


@api_bp.delete("/datasets/<filename>")
def delete_dataset(filename: str):
    """Delete a dataset from data/raw/."""
    safe_name = Path(filename).name  # Prevent directory traversal
    data_dir = Path(current_app.config["DATA_DIRECTORY"])
    download_archive = Path.home() / "Downloads" / "Compressed" / "archive"

    target = None
    if (data_dir / safe_name).is_file():
        target = data_dir / safe_name
    elif (download_archive / safe_name).is_file():
        target = download_archive / safe_name

    if not target or not target.is_file():
        return jsonify({"error": f"Dataset '{safe_name}' not found on server."}), 404

    target.unlink(missing_ok=True)

    # Unconditionally evict cached dataset service from memory so old data is never retained in RAM
    current_app.extensions.pop("dataset_service", None)
    current_app.config["DATASET_PATH"] = os.getenv("DATASET_PATH", "")

    # Switch to the next available dataset if one exists
    available = list_available_datasets()
    if available:
        current_app.config["DATASET_PATH"] = available[0]["path"]
        current_app.extensions.pop("dataset_service", None)

    try:
        active_ds = get_dataset()
        schema = active_ds.schema()
    except Exception:
        schema = None

    return jsonify({
        "message": f"Dataset '{safe_name}' deleted.",
        "datasets": list_available_datasets(),
        "dataset": schema
    })


@api_bp.post("/upload")
def upload():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Select a CSV or Excel dataset to upload."}), 400
    ext = Path(uploaded.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        return jsonify({"error": "Only CSV (.csv) and Excel (.xlsx, .xls) files are supported."}), 400

    filename = secure_filename(uploaded.filename)
    if not filename:
        filename = f"dataset_{int(time.time())}{ext}"

    destination = Path(current_app.config["DATA_DIRECTORY"]) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(destination)

    # Activate the uploaded file
    current_app.config["DATASET_PATH"] = str(destination)
    current_app.extensions.pop("dataset_service", None)

    try:
        dataset = get_dataset()
        return jsonify({
            "message": f"Dataset '{filename}' uploaded and activated.",
            "dataset": dataset.schema(),
            "datasets": list_available_datasets(),
            "active": filename
        })
    except DatasetUnavailable as error:
        destination.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 400


@api_bp.post("/reset-dataset")
def reset_dataset():
    """Reset configured dataset path back to default."""
    current_app.config["DATASET_PATH"] = os.getenv("DATASET_PATH", "")
    current_app.extensions.pop("dataset_service", None)

    try:
        dataset = get_dataset()
        return jsonify({
            "message": "Reverted to default dataset.",
            "dataset": dataset.schema(),
            "datasets": list_available_datasets(),
        })
    except DatasetUnavailable:
        return jsonify({
            "message": "No active dataset found.",
            "dataset": None,
            "datasets": list_available_datasets(),
        })


