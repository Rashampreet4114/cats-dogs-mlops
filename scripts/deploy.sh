#!/usr/bin/env bash
# Local deploy: (re)build/pull the image and (re)start the Docker Compose service.
# Used both manually and by the CD job on a self-hosted GitHub Actions runner.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${PULL_FROM_REGISTRY:-false}" = "true" ]; then
  IMAGE="${IMAGE:?Set IMAGE=ghcr.io/<owner>/cats-dogs-api:latest}"
  docker pull "$IMAGE"
  IMAGE="$IMAGE" docker compose up -d --force-recreate --no-build
else
  docker compose up -d --build
fi

echo "Waiting for the API to become healthy ..."
for i in $(seq 1 15); do
  if curl -s -o /dev/null http://localhost:8000/health; then
    echo "API is up."
    exit 0
  fi
  sleep 2
done

echo "API did not become healthy in time" >&2
exit 1
