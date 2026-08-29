# Cats vs Dogs — MLOps Pipeline

Binary image classification (Cats vs Dogs) for a pet-adoption platform, built as an
end-to-end MLOps pipeline: data versioning, experiment tracking, a containerized
FastAPI inference service, CI/CD, and basic monitoring.

> **Status**: the local pipeline (data → train → API → Docker) below is built and
> verified end-to-end. Git init, DVC remote, GitHub Actions CI/CD, and the
> Docker-Compose/self-hosted-runner deployment are **not set up yet** — they come in
> a follow-up pass once you've confirmed everything here works on your machine.

## Architecture at a glance

```
Kaggle dataset --> scripts/download_data.py --> data/raw/{cat,dog}
                                                     |
                                     src/data_processing.py (80/10/10 split, augment)
                                                     v
                                          data/processed/{train,val,test}
                                                     |
                                              src/train.py (PyTorch CNN)
                                                     |
                                    MLflow (mlruns/) <-+-> models/model.pt + metadata.json
                                                     |
                                            api/main.py (FastAPI)
                                      /health  /predict  /metrics
                                                     |
                                              Dockerfile / docker-compose.yml
```

## Manual walkthrough checklist (with visual confirmations)

Follow section 1→8 below in order. Each has a **✅ You should see** line — those are
the exact moments worth screenshotting/recording for the assignment's demo video and
submission evidence. Quick summary of what to capture:

| # | Step | Visual proof |
|---|------|--------------|
| 1 | Install deps | `pip install` ends with no errors |
| 2 | Get + split data | folder counts printed (`{'train': N, 'val': N, 'test': N}`) |
| 3 | Train | live epoch loss/accuracy in terminal + MLflow UI screenshot (metrics, confusion matrix, loss curve) |
| 4 | Unit tests | `pytest -v` green `PASSED` list |
| 5 | Run API (no Docker) | Swagger UI at `/docs`, a live `/predict` call with an uploaded photo |
| 6 | Docker | `docker compose ps` showing `healthy`, same `/predict` call against the container |
| 7 | Smoke test | terminal showing `Smoke tests passed.` |
| 8 | Post-deploy tracking | `reports/post_deploy_metrics.json` accuracy number |

## Prerequisites

- Python 3.11 (this project pins deps for 3.11)
- Docker Desktop (for containerization/local deploy testing)
- A free [Kaggle](https://www.kaggle.com) account + API token (only needed to pull the
  real dataset — you can smoke-test the whole pipeline without it, see below)

## 1. Create the virtual environment and install dependencies

```bash
cd cats-dogs-mlops
python3 -m venv mlops
source mlops/bin/activate        # Windows: mlops\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls in `requirements-api.txt` (the runtime-only deps also used
inside the Docker image) plus training/dev/test tooling (MLflow, scikit-learn,
matplotlib, pytest, kaggle, dvc, ...). All versions are pinned for reproducibility.

Or use the provided `Makefile`:

```bash
make install
```

**✅ You should see**: pip finishes with `Successfully installed ...` and no red error
text. Confirm with:

```bash
python -c "import torch, fastapi, mlflow; print('OK:', torch.__version__, fastapi.__version__, mlflow.__version__)"
```
which should print version numbers with no traceback.

## 2. Get the data

### Option A — real Kaggle dataset (recommended before a real training run)

1. Create a free Kaggle account, go to **Account Settings → API → Create New Token**.
   This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`
   (or export `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars instead).
3. Download a fast, laptop-friendly subset (2,000 images/class by default):

   ```bash
   python scripts/download_data.py
   # or the full ~25k-image dataset:
   python scripts/download_data.py --full
   ```

   This populates `data/raw/cat/` and `data/raw/dog/`.

**✅ You should see**: `Done. Images written to .../data/raw: {'cat': 1500, 'dog': 1500}`
(or your chosen count). Confirm visually:

```bash
ls data/raw/cat | wc -l
ls data/raw/dog | wc -l
open data/raw/cat/$(ls data/raw/cat | head -1)   # macOS: opens a sample image in Preview
```

### Option B — synthetic smoke-test data (no Kaggle account needed)

To sanity-check the whole pipeline (splits → training → API → Docker) without
waiting on a download, generate a tiny synthetic dataset instead:

```bash
python scripts/generate_sample_data.py --per-class 60
```

This is **not real training data** — accuracy numbers from it are meaningless. It only
exists to verify the pipeline wiring. Use Option A for anything you'll actually report.

### Split into train/val/test (80/10/10)

```bash
python -c "from src.data_processing import create_splits; print(create_splits('data/raw', 'data/processed'))"
# or: make split-data
```

This resizes nothing yet (that happens at load time) — it just partitions files into
`data/processed/{train,val,test}/{cat,dog}/`. Resizing to 224x224, normalization, and
augmentation (random flip/rotation/color-jitter for the train split) are applied by
`src/data_processing.py` when the PyTorch `Dataset` loads each image.

**✅ You should see**: a printed dict like `{'train': 2400, 'val': 300, 'test': 300}`.
Confirm visually:

### Dataset versioning with DVC

`data/raw` and `data/processed` are tracked with [DVC](https://dvc.org) instead of
being committed to git directly — git only stores small `.dvc` pointer files
(`data/raw.dvc`, `data/processed.dvc`, a hash + file count), while the actual images
live in a **local, no-cost DVC remote** at `../dvc-storage` (a sibling folder outside
this repo, configured in `.dvc/config`).

After downloading/splitting data yourself, re-track it and push to the local remote:

```bash
dvc add data/raw data/processed
dvc push
git add data/raw.dvc data/processed.dvc
git commit -m "Update dataset"
```

To reproduce someone else's checkout (or restore data after a fresh clone) instead of
re-downloading from Kaggle:

```bash
dvc pull   # restores data/raw and data/processed from ../dvc-storage using the .dvc files
```

**✅ You should see**: `dvc push`/`dvc pull` reporting a file count (e.g.
`9974 files pushed`), and `git status` showing only the small `.dvc` files changed —
never the actual image files.

```bash
find data/processed -mindepth 2 -maxdepth 2 -type d | sort
# should list data/processed/train/cat, .../train/dog, .../val/cat, .../val/dog, .../test/cat, .../test/dog
```

## 3. Train the model + track the experiment

```bash
python -m src.train --epochs 5 --batch-size 32
# or: make train
```

This trains `src/model.py`'s `SimpleCNN` (a baseline 4-conv-block CNN), and:
- logs params/metrics per epoch to **MLflow** (local file store at `mlruns/`)
- saves a confusion matrix and loss-curve plot to `reports/` (also logged as MLflow
  artifacts)
- saves the trained weights to `models/model.pt` and run metadata to
  `models/metadata.json`

**✅ You should see**: one line per epoch in the terminal —
`epoch 3/6 - train_loss=0.64 val_loss=0.66 val_acc=0.61` — ending with a
`test_loss=... test_acc=...` line. Then confirm the artifacts landed:

```bash
ls models/         # model.pt, metadata.json
ls reports/        # confusion_matrix.png, loss_curve.png
cat models/metadata.json
```

Browse the experiment runs (the main screenshot-worthy step for M1):

```bash
mlflow ui --backend-store-uri file:./mlruns
# open http://localhost:5000
```

**✅ You should see**: the MLflow UI listing your run under experiment
`cats-dogs-cnn`; click into it to see logged params (epochs, batch_size, lr),
per-epoch metrics charts (train_loss/val_loss/val_accuracy), and the
`confusion_matrix.png` / `loss_curve.png` artifacts attached to the run.

## 4. Run the unit tests

```bash
pytest tests/ -v
# or: make test
```

Covers: the image-preprocessing function (`test_data_processing.py`), the shared
inference utility (`test_predict.py`), and the FastAPI endpoints (`test_api.py`,
using an untrained checkpoint so it doesn't depend on a trained model existing).

**✅ You should see**: `pytest`'s green summary line, e.g.
`======= 7 passed in 3.04s =======`, with every test showing `PASSED`.

## 5. Run the API locally (no Docker)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or: make api
```

Verify it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/cat/<some_file>.jpg;type=image/jpeg"
curl http://localhost:8000/metrics
```

`/health` reports whether a model is loaded. `/predict` accepts a multipart image
upload and returns `{"label": "cat"|"dog", "probabilities": {"cat": ..., "dog": ...}}`.
`/metrics` exposes Prometheus-format counters (`http_requests_total`) and a latency
histogram (`http_request_latency_seconds`). Every request is also logged (method,
path, status, latency — no image bytes) to `logs/api.log`.

**✅ You should see**: `{"status":"ok","model_loaded":true}` from `/health`, and
`{"label":"cat"|"dog","probabilities":{"cat":...,"dog":...}}` from `/predict`. For the
best screenshot/recording moment, open **http://localhost:8000/docs** in a browser —
FastAPI's interactive Swagger UI — expand `POST /predict`, click "Try it out", upload
a real photo, click "Execute", and show the response body live. Then:

```bash
tail -5 logs/api.log     # confirms request logging (no image bytes, just metadata)
```

## 6. Build and run the container locally

```bash
docker build -t cats-dogs-api:local .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" cats-dogs-api:local
```

Or with Docker Compose (also mounts `logs/` so you can tail them from the host):

```bash
docker compose up -d --build
docker compose ps          # should show "healthy"
curl http://localhost:8000/health
docker compose down
```

Or via `make`:

```bash
make build   # docker build
make up      # docker compose up -d --build
make down    # docker compose down
```

**✅ You should see**: `docker compose ps` reporting `Up X seconds (healthy)`. Then
repeat the same `/docs` Swagger check from Step 5 against the containerized service —
same URL (`http://localhost:8000/docs`), now served from inside Docker instead of your
venv. That side-by-side ("works locally, works identically in the container") is good
demo-recording material.

## 7. Run the smoke test

Simulates what a CD pipeline should check post-deploy — fails (non-zero exit) if
either endpoint doesn't respond correctly:

```bash
bash scripts/smoke_test.sh http://localhost:8000
# or: make smoke
```

**✅ You should see**: `OK: /health`, `OK: /predict -> {...}`, and finally
`Smoke tests passed.` A non-zero exit / any `FAIL:` line means the CD pipeline should
be treated as broken.

## 8. Post-deployment performance tracking

Sends a batch of held-out test images (with known true labels) to a *running*
instance of the API, and writes accuracy + per-image results to
`reports/post_deploy_metrics.json`:

```bash
python scripts/track_performance.py --api-url http://localhost:8000 --max-per-class 50
# or: make track
```

**✅ You should see**: a terminal line like
`Sent 100 requests, accuracy=0.6 -> .../reports/post_deploy_metrics.json`. Then:

```bash
cat reports/post_deploy_metrics.json | python -m json.tool | head -20
```
shows per-image true label vs. predicted label + probabilities, plus the aggregate
`accuracy` field at the top — the M5 "model performance tracking" evidence.

## Project layout

```
src/                  data preprocessing, model, training, shared inference utility
api/                  FastAPI service (health / predict / metrics)
tests/                pytest unit + API tests
scripts/              download_data, generate_sample_data, deploy, smoke_test, track_performance
data/raw/             raw Kaggle images (data/raw/cat, data/raw/dog) — not committed
data/processed/       80/10/10 split (train/val/test) — not committed
models/               model.pt + metadata.json (trained artifact — committed)
reports/              confusion_matrix.png, loss_curve.png, post_deploy_metrics.json
mlruns/                MLflow local tracking store (generated by training)
logs/                 api.log (request/response logging)
Dockerfile, docker-compose.yml, requirements*.txt, Makefile
```

## CI/CD

The full pipeline is live at `.github/workflows/ci-cd.yml` on
[github.com/Rashampreet4114/cats-dogs-mlops](https://github.com/Rashampreet4114/cats-dogs-mlops)
(private repo — see note on visibility below):

- **`test`** (every push/PR, GitHub-hosted runner): checkout → install deps → `pytest`
- **`build`** (push/PR, GitHub-hosted runner): builds a multi-arch (amd64+arm64) Docker
  image; pushes to `ghcr.io/rashampreet4114/cats-dogs-mlops` (`:latest` + `:<sha>`) only
  on a push to `main`
- **`deploy`** (push to `main` only, **self-hosted runner** = this laptop): pulls the
  new image from GHCR, redeploys via `docker compose`, then runs
  `scripts/smoke_test.sh` — the job (and pipeline) fails if the smoke test fails

**Why self-hosted, and why it's safe**: Docker Compose deploys to this machine, not a
public server, so GitHub's cloud runners can't reach it to redeploy - a self-hosted
runner registered on this laptop is the only free way to make "push to `main` -> auto
redeploy" genuinely automatic. Self-hosted runners on a *public* repo are risky (a
fork's pull request could execute arbitrary code on the runner), so this repo is kept
**private**, and independently the `deploy` job is gated to `if: github.ref ==
'refs/heads/main' && github.event_name == 'push'` — never `pull_request` — so it could
never be reached by a fork's PR even if visibility changed later.

To run the runner yourself: register it once at
`Settings -> Actions -> Runners -> New self-hosted runner` on the repo, then from the
extracted runner folder:

```bash
./run.sh
```

Leave that terminal open while you want CD to be live; `Ctrl+C` stops it (and the
runner simply won't pick up new `deploy` jobs until it's running again — nothing else
is affected).

## What's next

All 5 assignment modules (M1-M5) are implemented and verified end-to-end. Remaining
work is packaging, not building:

- Package the submission zip (source, DVC/CI-CD/Docker configs, trained model artifacts)
- Record the <5 minute demo video (code change -> CI -> CD -> deployed prediction)
