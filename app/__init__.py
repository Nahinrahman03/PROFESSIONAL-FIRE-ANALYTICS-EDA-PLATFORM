"""Application factory for FireScope."""
from pathlib import Path
from flask import Flask, jsonify
from .config.settings import Config
from .routes.dashboard import dashboard_bp
from .routes.api import api_bp
from .routes.ml import ml_bp
from .routes.alerts import alerts_bp


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Ensure required runtime directories exist
    data_dir = app.config.get("DATA_DIRECTORY")
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(app.root_path).parent / "data" / "models").mkdir(parents=True, exist_ok=True)

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(ml_bp, url_prefix="/api/ml")
    app.register_blueprint(alerts_bp, url_prefix="/api/alerts")

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "The requested resource was not found."}), 404

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Unhandled application error: %s", error)
        return jsonify({"error": "An unexpected server error occurred."}), 500

    return app
