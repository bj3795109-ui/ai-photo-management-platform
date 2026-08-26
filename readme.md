# AI Photo Management Platform

An AI-powered photo management platform that connects to Google Photos and local storage, identifies duplicate and near-duplicate images, categorizes photos, groups people using facial recognition, and supports natural language search — built with FastAPI, PostgreSQL + pgvector, OpenCLIP, and Tesseract OCR, with a Vue-based web interface.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system architecture diagrams and detailed design decisions.

## Features

- **Web UI** — gallery with infinite scroll, natural language search, face clusters, duplicates review, and upload/import (local device, Google Photos, Google Drive)
- **Ingestion** from local storage and Google Photos (via the Picker API)
- **Natural language search** over your photo library using CLIP embeddings
- **Duplicate detection** — both exact/near-exact (perceptual hash) and semantic near-duplicates (CLIP similarity)
- **Face grouping** — detects and clusters faces across your library, similar to modern photo apps
- **Two-path categorization**:
  - Text-heavy images (documents, prescriptions, receipts) classified via **OCR + keyword matching**
  - Visual images (people, pets, travel, food, nature, vehicles, screenshots) classified via **CLIP zero-shot classification**
- **Bulk ingestion** with batched CLIP inference for large libraries
- **Background job queue** — `/dedup`, `/faces/cluster`, and `/faces/reconcile` run on Celery workers over Redis, not the request thread, with live progress tracking
- **Periodic face-cluster reconciliation** — a scheduled batch pass that re-validates and splits clusters where incremental matching has drifted
- Fully containerized with Docker

## Tech Stack

- **API**: FastAPI
- **Task queue**: Celery + Redis (worker + beat scheduler)
- **Frontend**: Vue 3 (single-file, CDN-loaded) + Tailwind
- **Database**: PostgreSQL + pgvector (vector similarity search)
- **Embeddings**: OpenCLIP (`ViT-B-32-quickgelu`, OpenAI weights)
- **Document classification**: Tesseract OCR (`pytesseract`) + keyword matching
- **Face detection/clustering**: `face_recognition` (HOG or CNN, `FACE_DETECTION_MODEL`) + DBSCAN (incremental matching + periodic batch reconciliation)
- **Storage**: Pluggable backend — local filesystem or Google Drive
- **Photo import**: Google Photos Picker API
- **Deployment**: Docker + docker-compose

## Setup

### Prerequisites

- Python 3.11+ (3.13 tested)
- Docker Desktop
- Tesseract OCR (see below)
- A Google Cloud project (for Google Drive storage and/or Google Photos import)

### 1. Clone and install dependencies

```bash
git clone https://github.com/bj3795109-ui/ai-photo-management-platform.git
cd ai-photo-management-platform
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

**Important (Windows):** `face_recognition` depends on the source `dlib` package, which fails to build without Visual Studio C++ Build Tools. This project uses `dlib-bin` (a prebuilt wheel) instead. After the main install, install `face_recognition` separately without pulling its `dlib` dependency:

```bash
pip install face_recognition --no-deps
```

### 2. Install Tesseract OCR

The `pytesseract` Python package is only a wrapper — it requires the actual Tesseract OCR **binary** to be installed separately.

- **Windows**: Download and run the installer from https://github.com/UB-Mannheim/tesseract/wiki (default install path `C:\Program Files\Tesseract-OCR\tesseract.exe`, which the code already looks for)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Docker**: already handled in the `Dockerfile` (installs `tesseract-ocr` via `apt-get`)

Verify it's working:
```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

### 3. Start PostgreSQL (with pgvector) and Redis

```bash
docker compose up -d db redis
```

### 4. Google Cloud setup (for Drive storage and/or Google Photos import)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Google Drive API** (for local-style storage) and/or the **Google Photos Picker API** (for importing from a user's Google Photos library)
3. Configure the OAuth consent screen (External, add yourself as a test user)
4. Create OAuth credentials (Desktop app type), download as `credentials.json`, place in the project root
5. On first run, `auth.py` will open a browser window for you to authorize the app; this generates `token.json`

**Note on Google Photos:** As of April 2025, Google restricted the Photos Library API — apps can no longer list a user's entire existing library without a formal Google verification process. This project uses the **Photos Picker API** instead: the user explicitly selects photos through Google's own picker UI each time, and the app receives access only to those selected items. See `ARCHITECTURE.md` for details.

**Note on token expiry:** While the Google Cloud OAuth app is in "Testing" publish status (the default), Google expires refresh tokens after 7 days — you'll need to re-authenticate periodically (a browser window will open automatically when this happens). Moving the project to "Production" status removes this limit.

### 5. Run the API, worker, and beat scheduler

```bash
python -m uvicorn api.main:app --reload
celery -A celery_app worker --loglevel=info      # in a second terminal
celery -A celery_app beat --loglevel=info         # in a third terminal
```

The worker executes `/dedup`, `/faces/cluster`, and `/faces/reconcile` jobs; beat triggers the periodic face-cluster reconciliation pass on a schedule (see "Background Jobs" below). The API itself works without them running, but those three endpoints will queue tasks that never get picked up.

Visit `http://127.0.0.1:8000/docs` for interactive API documentation (Swagger UI), or open `frontend/index.html` directly in a browser (or serve via `python -m http.server 8080` inside `frontend/`) for the web UI.

**If upgrading an existing installation or migrating data from Windows to Docker/Linux:**
Run the path normalization migration script to ensure all stored image paths use Linux-compatible `/` separators:
```bash
python -m scripts.fix_paths
```

**If Google Photos / Drive OAuth refresh token expires (7-day Google limit for apps in Testing mode):**
Run the re-authentication helper script on your host machine to refresh `token.json`:
```bash
python scripts/reauth.py
```
(The API also attempts this automatically on startup — safe to run either way, it's idempotent.)

### 6. Run with Docker (full stack)

```bash
docker compose up -d
```

This builds and runs all five containers together — `db`, `redis`, `api`, `worker`, and `beat`. Your project folder is bind-mounted into the API and worker containers, so local storage and any bulk-downloaded datasets remain accessible.

## Web UI

Open `frontend/index.html` in a browser (with the API running at `http://127.0.0.1:8000`). Includes:

- **Gallery** — infinite-scroll photo browsing
- **Search** — natural language search box
- **Upload & Import** — drag-and-drop local upload, Google Photos picker flow (with live status: idle → picking → importing → done/error), Google Drive
- **Faces** — browse detected face clusters
- **Duplicates** — review detected duplicate pairs side by side
- **Dashboard** — library stats and running job status

## API Usage

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and ingest a photo (multipart file upload) |
| GET | `/photos?offset=0&limit=50` | List photos, paginated, newest first |
| GET | `/photos/{id}` | Get photo metadata |
| GET | `/photos/{id}/image` | View/download the actual image file |
| DELETE | `/photos/{id}` | Delete a photo |
| GET | `/search?q=...&top_k=5` | Natural language search |
| POST | `/dedup` | Queue duplicate detection on a Celery worker (resumable, progress-tracked); returns a `task_id` |
| GET | `/duplicates?offset=0&limit=50` | List detected duplicate pairs, paginated, with viewable image URLs |
| POST | `/faces/cluster?model=hog\|cnn` | Queue face detection/clustering on a Celery worker (resumable, progress-tracked); returns a `task_id` |
| POST | `/faces/reconcile` | Manually trigger the periodic face-cluster reconciliation pass (also runs automatically via Celery beat) |
| GET | `/faces/clusters` | List face clusters, with viewable image URLs |
| POST | `/photos/{id}/categorize` | Categorize a photo (OCR document path, falls back to CLIP visual path) |
| GET | `/photos/{id}/categories` | Get a photo's assigned categories |
| GET | `/jobs/status` | Poll live progress of `/dedup`, `/faces/cluster`, or `/faces/reconcile` by job name |
| GET | `/jobs/{task_id}` | Poll a specific task by the id returned from the POST call |
| POST | `/google-photos/session` | Start a Google Photos picker session |
| POST | `/google-photos/session/{id}/import` | Import photos selected in a picker session |

### Bulk ingestion (large libraries)

For ingesting many images at once (faster than one-by-one via `/ingest`, uses batched CLIP inference):

```bash
python -m scripts.bulk_ingest /path/to/folder
```

## Testing

```bash
python -m pytest tests/ -v
```

15+ tests covering embedding, deduplication, face detection/clustering, categorization, and storage.

## Categorization: why two separate classification paths

Initial testing revealed that CLIP-only categorization was unreliable for the assignment's required text-based categories (`documents`, `prescriptions`, `receipts`). For example, searching "prescription" against a test library returned photos of people wearing **eyeglasses** — CLIP was matching on the strongest semantic association in its training data ("prescription" → "prescription glasses"), not on document content, because CLIP judges images by visual similarity and has no ability to read text. Prescriptions, receipts, and generic documents all look like "text on a light background" to it, so it cannot reliably tell them apart.

**Fix:** `pipeline/document_classifier.py` runs OCR (Tesseract) on the image first. If the image is text-heavy enough (a minimum word-count threshold), it classifies based on **keyword matching** against the extracted text (e.g. "rx", "dosage", "pharmacy" → prescriptions; "total", "subtotal", "tax" → receipts; otherwise → documents). Only if the image doesn't have enough text does it fall back to the original CLIP visual classification path for the remaining categories (people, pets, travel, food, nature, vehicles, screenshots, other).

**Validated against the Tobacco3482 dataset** (scanned business/legal documents — memos, letters, forms) after the fix: photos correctly classified as `documents` based on actual OCR'd content, rather than random CLIP guesses.

## Known Limitations

- **No GPU acceleration** — CLIP and face embedding inference run on CPU.
- **Google Drive backend** has no automated retry/error handling beyond what the Drive API raises natively.
- **Google Photos** integration only provides access to photos the user actively selects in each picker session — there's no way to browse/search a user's full library from within the app before they've picked (a platform-level restriction, not a limitation of this implementation — see `ARCHITECTURE.md`).
- **Document classification is keyword-based, not ML-based.** This is transparent and debuggable, and works well for clearly-worded documents, but can misclassify text that doesn't contain expected keywords (e.g. a receipt photographed at an angle with OCR errors, or a prescription in a format that doesn't use common terms). A production system would likely combine this with a trained text classifier for better recall.
- **OCR quality depends on image quality.** Blurry, low-resolution, or heavily skewed photos of documents will produce poor OCR text, which can cause the word-count threshold to fail and incorrectly fall through to the CLIP visual path.
- **Database migrations are handled with inline/manual scripts, not a migration tool.** The `created_at` column addition uses an idempotent `DO $$ ... END $$` block run on startup, plus a standalone script (`scripts/add_created_at.py`) doing the same thing. This works for a single-developer project at this stage but doesn't track migration history or support rollback — a production system would use Alembic instead.

## Background Jobs (Celery + Redis)

`/dedup`, `/faces/cluster`, and `/faces/reconcile` no longer run on the request thread — they're dispatched to a Celery worker over Redis and return immediately with a `task_id`. `docker-compose up` now runs five services: `db`, `redis`, `api`, `worker`, and `beat`.

- **`worker`** — executes queued jobs (`celery -A celery_app worker`). The long CPU-bound work (CLIP/face embedding comparisons, DBSCAN) is isolated from the API process, so a slow job no longer ties up a request thread or risks an HTTP timeout.
- **`beat`** — the Celery scheduler (`celery -A celery_app beat`). Only triggers tasks on a schedule; the actual work still runs on `worker`.
- Poll `GET /jobs/status` (by job name) or `GET /jobs/{task_id}` (by the id returned from the POST call) to watch progress — this still works exactly as before, since the worker writes to the same `job_status` table the request thread used to write to directly.
- Configure via `REDIS_URL` (defaults to `redis://localhost:6379/0`) and `RECONCILE_INTERVAL_SECONDS` (defaults to 6 hours).

## Face-Cluster Reconciliation

Incremental clustering (in `/faces/cluster`) matches each new face against only its single nearest known face — fast, and it means only new photos need processing on each run. But it has no correction mechanism: if two visually similar but distinct people are ever matched together early, everything loosely close to either of them can keep joining that same cluster, and nothing re-checks whether the cluster still makes sense as it grows. This was observed directly on the full 13,233-image LFW dataset — one cluster grew to 1,056 faces, verified visually to contain multiple different people.

**Fix:** `pipeline/jobs.py::run_reconciliation_job()` (wired to `tasks.reconcile_face_clusters`, scheduled via Celery beat and also triggerable manually via `POST /faces/reconcile`) periodically re-examines every `FaceCluster` above a size threshold (`RECONCILE_SIZE_THRESHOLD`, default 30). It re-clusters that cluster's own faces from scratch with batch DBSCAN at a stricter tolerance (`RECONCILE_TOLERANCE`, default 0.5) — `pipeline/faces.py::cluster_faces()`, the batch clustering function that existed but wasn't previously wired into the running system. If DBSCAN disagrees with the incremental result, the largest resulting group keeps the cluster's original id; every other group is split into a newly created cluster. Small/coherent clusters are left untouched.

This trades some CPU time (re-clustering large clusters periodically) for a real correction mechanism, rather than only detecting drift after the fact in a "Known Limitations" note.

## Face Detection Model

`FACE_DETECTION_MODEL` (env var, default `hog`) selects the face detector used by `pipeline/faces.py::detect_faces()`. `hog` is fast but less reliable on angled/profile faces; `cnn` is slower — meaningfully slower without a GPU — but more accurate on non-frontal faces. Override per-run via the `model` query param on `POST /faces/cluster` (e.g. `POST /faces/cluster?model=cnn`).



## Scaling Considerations (tested up to 116,000+ images)

The system was stress-tested well beyond toy-scale demos, combining the full LFW dataset (13,233 images), STL-10's unlabeled split (100,000 images), the Tobacco3482 document dataset, and Google Photos imports — **116,000+ total images**.

### What scaled well

- **Bulk ingestion** (`scripts/bulk_ingest.py`) — batches 32 images per CLIP forward pass. Ingested 13,233 images in ~458 seconds, and 100,000 images in ~3,876 seconds (~26 images/sec), comfortably CPU-bound and demo-viable.
- **Search** (`/search`) — backed by a pgvector `ivfflat` index on `photos.embedding`, so queries return in milliseconds regardless of library size.
- **Face clustering re-runs** — a `faces_processed` flag per photo ensures only new/unprocessed photos are touched on subsequent calls.
- **Gallery browsing** — paginated with infinite scroll rather than loading the entire library at once, keeping the UI responsive regardless of total photo count.

### What required rework at this scale

Both `/dedup` and `/faces/cluster` were initially implemented as long-running, single-transaction endpoints that only committed once at the very end, executed inline on the request thread — discovered to be a real problem at 100k+ scale (long waits with no visibility, any interruption meant losing all progress, and a slow request risked an HTTP timeout). This was fixed with:

1. **Periodic commits** — every 2,000 photos for `/dedup`, every 500 for `/faces/cluster` — instead of one commit at the end
2. **Resume flags** — a `dedup_processed` flag (mirroring the existing `faces_processed` pattern) so a restart skips already-completed photos instead of re-scanning the whole library
3. **A `job_status` table + `GET /jobs/status` endpoint** — written on each periodic commit, pollable from a separate request while the job is still running, so progress is visible in real time instead of a silent multi-hour wait
4. **Reduced query count in `/dedup`** — the original version issued an extra existence-check database query per potential duplicate match found; this was replaced with an in-memory lookup set built once at the start, cutting total query volume significantly
5. **Moved off the request thread entirely** — both endpoints now dispatch to a Celery worker over Redis (see "Background Jobs" below) instead of running inline, so a multi-hour job no longer occupies an API request or risks a gateway timeout

This was discovered through direct experience: the initial `/dedup` and `/faces/cluster` runs against the full 113k library were interrupted (laptop sleep, accidental terminal close) partway through, which under the original one-commit-at-the-end design would have meant restarting from zero each time. The fixes above make both operations genuinely production-viable at this scale rather than a fragile, unmonitorable batch job.

## Roadmap

- [ ] ML-based document classifier to complement/replace keyword matching
- [ ] Alembic migrations instead of inline/manual scripts
- [ ] Postman collection
- [ ] Cap cluster-size threshold dynamically / auto-tune `RECONCILE_TOLERANCE` instead of a fixed default
