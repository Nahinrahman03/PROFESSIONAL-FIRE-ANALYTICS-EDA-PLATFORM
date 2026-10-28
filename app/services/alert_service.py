"""Alert rule persistence, evaluation, and optional email notification."""
from __future__ import annotations

import json
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
import logging
from typing import Any

from app.config.settings import Config

logger = logging.getLogger(__name__)


def _get_config(key: str, default=None):
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            return current_app.config.get(key, default)
    except Exception:
        pass
    return getattr(Config, key, default)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _alerts_path() -> Path:
    return Path(_get_config("ALERTS_FILE", str(Config.PROJECT_ROOT / "data" / "alerts.json")))


def _load_rules() -> list[dict]:
    p = _alerts_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_rules(rules: list[dict]) -> None:
    p = _alerts_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rules, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_rules() -> list[dict]:
    """Return all alert rules, newest first."""
    return sorted(_load_rules(), key=lambda r: r.get("created_at", ""), reverse=True)


def get_rule(rule_id: str) -> dict | None:
    return next((r for r in _load_rules() if r["id"] == rule_id), None)


def create_rule(data: dict) -> dict:
    """Validate and persist a new alert rule; return the created rule."""
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("Alert name is required.")

    rule: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "confidence": str(data.get("confidence", "")).strip(),
        "source": str(data.get("source", "")).strip(),
        "daynight": str(data.get("daynight", "")).strip(),
        "frp_min": _opt_float(data.get("frp_min")),
        "frp_max": _opt_float(data.get("frp_max")),
        "brightness_min": _opt_float(data.get("brightness_min")),
        "email": str(data.get("email", "")).strip(),
        "active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_fired": None,
        "last_count": 0,
    }
    rules = _load_rules()
    rules.append(rule)
    _save_rules(rules)
    return rule


def delete_rule(rule_id: str) -> bool:
    rules = _load_rules()
    new_rules = [r for r in rules if r["id"] != rule_id]
    if len(new_rules) == len(rules):
        return False
    _save_rules(new_rules)
    return True


def toggle_rule(rule_id: str) -> dict | None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule["active"] = not rule.get("active", True)
            _save_rules(rules)
            return rule
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rules(dataset) -> list[dict]:
    """
    Evaluate all active alert rules against *dataset* (a DatasetService).
    Returns a list of firing alert dicts with a ``matched_count`` field.
    """
    rules = _load_rules()
    active = [r for r in rules if r.get("active", True)]
    if not active:
        return []

    firing: list[dict] = []
    updated = False

    for rule in rules:
        if not rule.get("active", True):
            continue

        try:
            count = _evaluate_single(rule, dataset)
        except Exception:  # noqa: BLE001 — never crash the check endpoint
            count = 0

        if count > 0:
            firing_rule = dict(rule)
            firing_rule["matched_count"] = count

            # Track first-fire for email (only send once per new firing event)
            last_fired_before = rule.get("last_fired")
            rule["last_fired"] = datetime.now(timezone.utc).isoformat()
            rule["last_count"] = count
            updated = True

            # Send email only if email is configured AND this is a new firing
            if rule.get("email") and not last_fired_before:
                _send_email(rule, count)

            firing.append(firing_rule)

    if updated:
        _save_rules(rules)

    return firing


def _evaluate_single(rule: dict, dataset) -> int:
    """Apply rule conditions to the dataset and return the matching record count."""
    params: dict[str, Any] = {}

    if rule.get("confidence"):
        params["confidence"] = rule["confidence"]
    if rule.get("source"):
        params["source"] = rule["source"]
    if rule.get("daynight"):
        params["daynight"] = rule["daynight"]
    if rule.get("frp_min") is not None:
        params["frp_min"] = rule["frp_min"]
    if rule.get("frp_max") is not None:
        params["frp_max"] = rule["frp_max"]
    if rule.get("brightness_min") is not None:
        params["brightness_min"] = rule["brightness_min"]

    frame = dataset.filter(params)
    return int(len(frame))


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

def _send_email(rule: dict, count: int) -> None:
    """Send a plain-text alert email if SMTP is configured. Silently skips if not."""
    server = _get_config("MAIL_SERVER", "")
    if not server:
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"🔥 Fire Alert Triggered: {rule['name']}"
        msg["From"] = _get_config("MAIL_FROM") or _get_config("MAIL_USERNAME", "")
        msg["To"] = rule["email"]

        conditions = []
        if rule.get("confidence"):
            conditions.append(f"Confidence = {rule['confidence']}")
        if rule.get("source"):
            conditions.append(f"Source = {rule['source']}")
        if rule.get("daynight"):
            conditions.append(f"Day/Night = {rule['daynight']}")
        if rule.get("frp_min") is not None:
            conditions.append(f"FRP ≥ {rule['frp_min']}")
        if rule.get("frp_max") is not None:
            conditions.append(f"FRP ≤ {rule['frp_max']}")
        if rule.get("brightness_min") is not None:
            conditions.append(f"Brightness ≥ {rule['brightness_min']}")

        body = (
            f"Your fire alert '{rule['name']}' has triggered.\n\n"
            f"Matched records : {count:,}\n"
            f"Conditions      : {', '.join(conditions) or 'None'}\n"
            f"Triggered at    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            "-- FireScope EDA Platform"
        )
        msg.set_content(body)

        port = int(_get_config("MAIL_PORT", 587))
        username = _get_config("MAIL_USERNAME", "")
        password = _get_config("MAIL_PASSWORD", "")

        with smtplib.SMTP(server, port, timeout=10) as smtp:
            smtp.ehlo()
            if port != 465:
                smtp.starttls()
            if username:
                smtp.login(username, password)
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alert email failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
