import logging
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import HealthResponse, PredictResponse, VersionResponse
from src.predict import load_model, predict_bytes

APP_VERSION = "1.1.0"

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "model.pt"))
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cats_dogs_api")
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=1_000_000, backupCount=3)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "HTTP request latency (seconds)", ["path"]
)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if Path(MODEL_PATH).exists():
        _model = load_model(MODEL_PATH)
        logger.info("model loaded from %s", MODEL_PATH)
    else:
        logger.warning("model file not found at %s - /predict will return 503", MODEL_PATH)
    yield


app = FastAPI(title="Cats vs Dogs Inference API", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    path = request.url.path
    REQUEST_COUNT.labels(method=request.method, path=path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(path=path).observe(duration)

    # No request/response bodies (e.g. image bytes) are logged, only metadata.
    logger.info(
        "%s %s status=%s duration_ms=%.1f",
        request.method,
        path,
        response.status_code,
        duration * 1000,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/version", response_model=VersionResponse)
def version():
    return VersionResponse(version=APP_VERSION, model_loaded=_model is not None)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        result = predict_bytes(_model, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

    return PredictResponse(**result)
