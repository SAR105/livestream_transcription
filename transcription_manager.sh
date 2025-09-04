#!/usr/bin/env bash
set -euo pipefail

ENTRYPOINT="transcription.py"           
CONDA_ENV="multiscreen"          
PID_FILE="server.pid"         
OUT_LOG="server.out"          
ERR_LOG="server.err"          

CONDA_BIN="$HOME/miniconda3/bin/conda"

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start() {
  if is_running; then
    echo "Server already running (PID $(cat $PID_FILE))."
    exit 0
  fi
  echo "Starting server..."
  nohup "$CONDA_BIN" run -n "$CONDA_ENV" python -u "$ENTRYPOINT" \
    >>"$OUT_LOG" 2>>"$ERR_LOG" &
  echo $! > "$PID_FILE"
  echo "Started (PID $(cat $PID_FILE))"
}

stop() {
  if ! is_running; then
    echo "Server not running."
    rm -f "$PID_FILE"
    exit 0
  fi
  pid=$(cat "$PID_FILE")
  echo "Stopping PID $pid..."
  kill "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped."
}

restart() {
  stop || true
  start
}

status() {
  if is_running; then
    echo "Running (PID $(cat $PID_FILE))"
  else
    echo "Not running"
  fi
}

logs() {
  tail -f "$OUT_LOG" "$ERR_LOG"
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) restart ;;
  status)  status ;;
  logs)    logs ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" ;;
esac
