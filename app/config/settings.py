from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    PROJECT_ROOT = PROJECT_ROOT
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-change-me")
    DEBUG = os.getenv("FLASK_ENV", "production").lower() == "development"
    DATASET_PATH = os.getenv("DATASET_PATH", "")
    DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
    MAP_POINT_LIMIT = int(os.getenv("MAP_POINT_LIMIT", "5000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    JSON_SORT_KEYS = False

    # Alerting — persistence
    ALERTS_FILE = os.getenv("ALERTS_FILE", str(PROJECT_ROOT / "data" / "alerts.json"))

    # Alerting — optional SMTP email delivery
    MAIL_SERVER = os.getenv("MAIL_SERVER", "")
    MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
    MAIL_FROM = os.getenv("MAIL_FROM", "")

