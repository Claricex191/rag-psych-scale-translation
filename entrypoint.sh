#!/usr/bin/env bash
set -e

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"

# Distinct paths per server — each server reads QDRANT_PATH/GUIDELINES_DIR from
# its own process environment, set inline below (not a container-wide ENV),
# so forward and backward never collide on the same Qdrant collection dir.
QDRANT_PATH="$VOLUME_ROOT/qdrant-db-forward" \
GUIDELINES_DIR="$VOLUME_ROOT/guidelines-forward" \
python server_forward.py --host 127.0.0.1 --port 8000 &
FWD_PID=$!

QDRANT_PATH="$VOLUME_ROOT/qdrant-db-backward" \
GUIDELINES_DIR="$VOLUME_ROOT/guidelines-backward" \
python server_backward.py --host 127.0.0.1 --port 8001 &
BWD_PID=$!

trap 'kill $FWD_PID $BWD_PID $ST_PID 2>/dev/null' TERM INT

# Wait for a PID's /health to respond, but fail fast (instead of looping
# forever) if that process has already died — e.g. OOM-killed while loading
# its embedding model. Confirmed this happens on memory-constrained hosts.
wait_for_health() {
  local pid=$1 port=$2 name=$3
  echo "Waiting for $name (pid $pid, port $port) to come up..."
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name (pid $pid) exited before becoming healthy. Check logs above (likely OOM if 'Killed' appears)." >&2
      exit 1
    fi
    sleep 1
  done
  echo "$name is up."
}

wait_for_health "$FWD_PID" 8000 "server_forward.py"
wait_for_health "$BWD_PID" 8001 "server_backward.py"

# enableCORS/enableXsrfProtection=false: RunPod's proxy presents a different
# external hostname than Streamlit sees internally, which fails its default
# WebSocket Origin check and leaves the page stuck with a blank UI forever
# (confirmed via browser console: repeated "WebSocket onerror"). Safe to
# disable here since the password gate is the actual access control, not CORS.
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false --server.enableXsrfProtection false &
ST_PID=$!

# Exit (and let RunPod restart the pod) if any one process dies.
wait -n "$FWD_PID" "$BWD_PID" "$ST_PID"
exit 1
