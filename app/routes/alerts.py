"""Alerts blueprint — CRUD for alert rules and evaluation endpoint."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.services.alert_service import (
    create_rule,
    delete_rule,
    evaluate_rules,
    list_rules,
    toggle_rule,
)
from app.services.dataset_service import DatasetUnavailable, get_dataset

alerts_bp = Blueprint("alerts", __name__)


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@alerts_bp.errorhandler(ValueError)
def validation_error(error):
    return jsonify({"error": str(error)}), 400


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@alerts_bp.get("")
def get_alerts():
    """Return all saved alert rules."""
    return jsonify({"rules": list_rules()})


@alerts_bp.post("")
def post_alert():
    """Create a new alert rule."""
    data = request.get_json(silent=True) or {}
    rule = create_rule(data)
    return jsonify({"rule": rule, "message": "Alert rule created."}), 201


@alerts_bp.delete("/<rule_id>")
def remove_alert(rule_id: str):
    """Delete an alert rule by ID."""
    if delete_rule(rule_id):
        return jsonify({"message": "Alert deleted."})
    return jsonify({"error": "Alert not found."}), 404


@alerts_bp.patch("/<rule_id>/toggle")
def toggle_alert(rule_id: str):
    """Enable or disable an alert rule."""
    rule = toggle_rule(rule_id)
    if rule:
        status = "enabled" if rule["active"] else "disabled"
        return jsonify({"rule": rule, "message": f"Alert {status}."})
    return jsonify({"error": "Alert not found."}), 404


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------

@alerts_bp.post("/check")
def check_alerts():
    """
    Evaluate all active alert rules against the loaded dataset.
    Returns firing alerts with matched_count; safe to poll frequently.
    """
    try:
        dataset = get_dataset()
    except DatasetUnavailable:
        # No dataset loaded — no alerts can fire
        return jsonify({"firing": [], "total_firing": 0})

    firing = evaluate_rules(dataset)
    return jsonify({
        "firing": firing,
        "total_firing": len(firing),
    })
