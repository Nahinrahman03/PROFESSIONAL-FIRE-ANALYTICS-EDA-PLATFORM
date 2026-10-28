"""Tests for Streamlit platform integration and analytical pipelines."""
import pandas as pd
from app.services.dataset_service import DatasetService, get_dataset
from app.services.analytics_service import AnalyticsService
from app.services.alert_service import evaluate_rules, create_rule, list_rules, delete_rule
from app.ml.trainer import train_regression, train_clustering, train_classification


def test_standalone_dataset_and_analytics(tmp_path):
    path = tmp_path / "test_fires.csv"
    pd.DataFrame({
        "latitude": [12.0, 13.0, 14.0],
        "longitude": [78.0, 79.0, 80.0],
        "acq_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "frp": [3.5, 10.0, 25.0],
        "brightness": [310.0, 320.0, 335.0],
        "satellite": ["Aqua", "Terra", "Aqua"],
        "confidence": ["high", "nominal", "high"],
        "daynight": ["D", "N", "D"],
    }).to_csv(path, index=False)

    service = DatasetService(str(path))
    assert service.columns["latitude"] == "latitude"
    assert service.columns["frp"] == "frp"
    assert len(service.raw) == 3

    # Analytics service
    analytics = AnalyticsService(service)
    summary = analytics.summary(service.frame)
    assert summary["total_detections"] == 3
    assert len(summary["kpis"]) >= 4

    ts = analytics.time_series(service.frame, period="day")
    assert ts["available"] is True

    intensity = analytics.intensity(service.frame)
    assert intensity["available"] is True

    confidence = analytics.confidence_analysis(service.frame)
    assert confidence["available"] is True

    geo = analytics.geography(service.frame)
    assert geo["available"] is True


def test_ml_trainers_on_dataset(tmp_path):
    path = tmp_path / "ml_fires.csv"
    # Create sufficient rows for ML splitting
    rows = []
    for i in range(40):
        rows.append({
            "latitude": 10.0 + (i * 0.1),
            "longitude": 20.0 + (i * 0.1),
            "acq_date": f"2024-01-{(i % 28) + 1:02d}",
            "frp": 5.0 + (i * 1.5),
            "brightness": 300.0 + i,
            "satellite": "VIIRS" if i % 2 == 0 else "MODIS",
            "confidence": "high" if i % 3 == 0 else ("nominal" if i % 3 == 1 else "low"),
            "daynight": "D" if i % 2 == 0 else "N",
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    ds = DatasetService(str(path))
    detected = ds.columns

    # 1. Regression
    res_reg = train_regression(ds.frame, detected, n_estimators=10)
    assert res_reg["available"] is True
    assert "r2" in res_reg["metrics"]
    assert len(res_reg["feature_importances"]) > 0

    # 2. Clustering
    res_clu = train_clustering(ds.frame, detected, n_clusters=3)
    assert res_clu["available"] is True
    assert res_clu["n_clusters"] == 3

    # 3. Classification
    res_cls = train_classification(ds.frame, detected, n_estimators=10)
    assert res_cls["available"] is True
    assert "accuracy" in res_cls["metrics"]


def test_streamlit_views_render():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=35)
    assert len(at.exception) == 0

    # Test Fire Intensity page specifically (verifies trendline and stats)
    at.sidebar.radio[0].set_value("⚡ Fire Intensity (FRP)").run(timeout=35)
    assert len(at.exception) == 0


