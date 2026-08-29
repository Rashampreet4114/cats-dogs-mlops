#!/usr/bin/env bash
# Post-deploy smoke test: hits /health and /predict on a running instance.
# Exits non-zero (and should fail the CI/CD pipeline) if either check fails.
set -euo pipefail

API_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-}"

echo "Checking ${API_URL}/health ..."
HEALTH_STATUS=$(curl -s -o /tmp/health_response.json -w "%{http_code}" "${API_URL}/health")
if [ "$HEALTH_STATUS" != "200" ]; then
  echo "FAIL: /health returned status $HEALTH_STATUS"
  cat /tmp/health_response.json || true
  exit 1
fi
if ! grep -q '"status":"ok"' /tmp/health_response.json; then
  echo "FAIL: /health body did not report status=ok"
  cat /tmp/health_response.json
  exit 1
fi
echo "OK: /health"

if [ -z "$SAMPLE_IMAGE" ]; then
  # Fall back to any image under data/processed/test if one exists.
  SAMPLE_IMAGE=$(find data/processed/test -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | head -n 1 || true)
fi

if [ -z "$SAMPLE_IMAGE" ] || [ ! -f "$SAMPLE_IMAGE" ]; then
  echo "WARN: no sample image found to smoke-test /predict (skipping predict check)"
  exit 0
fi

echo "Checking ${API_URL}/predict with ${SAMPLE_IMAGE} ..."
PREDICT_STATUS=$(curl -s -o /tmp/predict_response.json -w "%{http_code}" \
  -X POST "${API_URL}/predict" -F "file=@${SAMPLE_IMAGE};type=image/jpeg")
if [ "$PREDICT_STATUS" != "200" ]; then
  echo "FAIL: /predict returned status $PREDICT_STATUS"
  cat /tmp/predict_response.json || true
  exit 1
fi
if ! grep -q '"label"' /tmp/predict_response.json; then
  echo "FAIL: /predict response missing 'label'"
  cat /tmp/predict_response.json
  exit 1
fi
echo "OK: /predict -> $(cat /tmp/predict_response.json)"

echo "Smoke tests passed."
