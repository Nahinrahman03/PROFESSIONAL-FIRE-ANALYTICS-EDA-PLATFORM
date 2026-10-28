import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Auto-fallback to project virtual environment if dependencies are missing
_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        import subprocess
        result = subprocess.run([str(_VENV_PYTHON)] + sys.argv, cwd=str(_PROJECT_ROOT))
        sys.exit(result.returncode)

import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
    port = os.getenv("PORT", "8501")
    host = os.getenv("HOST", "0.0.0.0")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        f"--server.address={host}",
    ] + sys.argv[1:]
    sys.exit(stcli.main())
