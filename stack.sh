#!/usr/bin/env bash
#
# stack.sh -- one-command control for the vLLM + Bifrost + Model Manager stack.
#
# Usage:
#   ./stack.sh up        Build images and start everything (detached)
#   ./stack.sh down      Stop and remove containers (keeps named volumes)
#   ./stack.sh restart   Full cycle: down, rebuild, up   <-- "stop and up + build"
#   ./stack.sh rebuild   down + build --no-cache + up (force clean image build)
#   ./stack.sh stop      Stop containers without removing them
#   ./stack.sh start     Start previously-stopped containers
#   ./stack.sh logs [svc]   Follow logs (optionally for one service)
#   ./stack.sh ps        Show container status
#   ./stack.sh pull      Pull latest upstream images (vllm, bifrost)
#
set -euo pipefail

# Always run from the repo root (the dir this script lives in), so it works
# no matter where it's invoked from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Support both Docker Compose v2 ("docker compose") and legacy ("docker-compose").
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "error: Docker Compose not found (need 'docker compose' or 'docker-compose')." >&2
  exit 1
fi

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

usage() { sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  up)
    log "Building images and starting the stack..."
    "${DC[@]}" up -d --build
    "${DC[@]}" ps
    log "Up. vLLM can take a few minutes to become healthy on first start."
    log "Follow startup with:  ./stack.sh logs vllm"
    ;;
  down)
    log "Stopping and removing containers (named volumes kept)..."
    "${DC[@]}" down
    ;;
  restart)
    log "Restarting: down -> rebuild -> up ..."
    "${DC[@]}" down
    "${DC[@]}" up -d --build
    "${DC[@]}" ps
    ;;
  rebuild)
    log "Clean rebuild: down -> build --no-cache -> up ..."
    "${DC[@]}" down
    "${DC[@]}" build --no-cache
    "${DC[@]}" up -d
    "${DC[@]}" ps
    ;;
  stop)
    "${DC[@]}" stop "$@"
    ;;
  start)
    "${DC[@]}" start "$@"
    "${DC[@]}" ps
    ;;
  logs)
    "${DC[@]}" logs -f --tail=100 "$@"
    ;;
  ps|status)
    "${DC[@]}" ps
    ;;
  pull)
    "${DC[@]}" pull
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "error: unknown command '$cmd'" >&2
    usage
    exit 1
    ;;
esac
