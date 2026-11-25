#!/usr/bin/env bash
# Start the Libras Recognizer: sets up .venv the first time, then runs the server.
#   ./run.sh              -> http://localhost:8000
#   PORT=8080 ./run.sh    -> another port
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.12 python3.13 python3.14 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then echo "Python 3.11+ not found. Install it (e.g. brew install python@3.12)."; exit 1; fi
  echo "Creating .venv with $PY ($($PY --version))"
  "$PY" -m venv .venv
fi

# (re)install dependencies when requirements.txt changes
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed
fi

echo ""
echo "  Starting the Libras Recognizer... (the first start can take up to a minute)"
echo "  Wait for the READY line, then open http://localhost:$PORT in Chrome."
echo "  Stop with Ctrl+C"
echo ""
export LIBRAS_PORT="$PORT"
if [ -n "${LIBRAS_VERBOSE:-}" ]; then
  exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
fi
# Hide MediaPipe's harmless C++ info/warning lines ("I0000 ...", "W0000 ...").
# Everything else (including real errors) still shows. LIBRAS_VERBOSE=1 shows all.
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
  2> >(trap '' INT; grep --line-buffered -vE '^[IW][0-9]{4} [0-9:.]+ +[0-9]+ [^ ]+\.cc:[0-9]+\]|^INFO: Created TensorFlow Lite' >&2)
