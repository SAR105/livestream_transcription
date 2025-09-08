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

  nohup "$CONDA_BIN" run -n "$CONDA_ENV" uvicorn live_ts:app \
    --host 127.0.0.1 --port 8888 \
    >>"$OUT_LOG" 2>>"$ERR_LOG" &

  echo "Starting server... Check on port 8888 to verify."
}

stop() {
  pida=$(ps -ef | grep "[m]ultiscreen" | awk '{print $2}')
  if [ -z "$pida" ]; then
    echo "Server not running."
    exit 0
  fi
  echo "Killing process $pida..."
  kill -9 "$pida"
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
