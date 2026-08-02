#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$ROOT/docker/docker-compose.yml"

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 <script.py> [args...]" >&2
  exit 1
fi

docker compose -f "$COMPOSE_FILE" --project-directory "$ROOT/docker" \
  run --rm --no-deps coral python3 "$@"
