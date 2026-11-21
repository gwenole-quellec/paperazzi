#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Create the virtual environment on first use.
if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

# State-aware tabs require Streamlit 1.55+. Upgrade an older existing
# environment when the user installs a new Paperazzi release over it.
if ! .venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    major, minor = map(int, version("streamlit").split(".")[:2])
except (PackageNotFoundError, ValueError):
    raise SystemExit(1)

raise SystemExit(0 if (major, minor) >= (1, 55) else 1)
PY
then
    .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/streamlit run app.py
