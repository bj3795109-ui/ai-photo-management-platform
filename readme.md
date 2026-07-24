# AI Photo Management Platform

An AI-powered photo management platform that connects to Google Photos and local storage, identifies duplicate and near-duplicate images, categorizes photos, groups people using facial recognition, and supports natural language search — built with FastAPI, PostgreSQL + pgvector, and OpenCLIP.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system architecture diagrams and detailed design decisions.

## Features

- **Ingestion** from local storage and Google Photos (via the Picker API)
- **Natural language search** over your photo library using CLIP embeddings
- **Duplicate detection** — both exact/near-exact (perceptual hash) and semantic near-duplicates (CLIP similarity)
- **Face grouping** — detects and clusters faces across your library, similar to modern photo apps
- **Zero-shot categorization** into: documents, prescriptions, receipts, people, travel, pets, food, screenshots, vehicles, nature, and other
- **Bulk ingestion** with batched CLIP inference for large libraries
- **Resumable long-running jobs** (dedup, face clustering) with live progress tracking
- Fully containerized with Docker

## Tech Stack

- **API**: FastAPI
- **Database**: PostgreSQL + pgvector (vector similarity search)
- **Embeddings**: OpenCLIP (`ViT-B-32-quickgelu`, OpenAI weights)
- **Face detection/clustering**: `face_recognition` (HOG) + DBSCAN / incremental nearest-neighbor matching
- **Storage**: Pluggable backend — local filesystem or Google Drive
- **Photo import**: Google Photos Picker API
- **Deployment**: Docker + docker-compose

## Setup

### Prerequisites

- Python 3.11+ (3.13 tested)
- Docker Desktop
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

### 2. Start PostgreSQL (with pgvector)

```bash
docker compose up -d db
```

### 3. Google Cloud setup (for Drive storage and/or Google Photos import)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Google Drive API** (for local-style storage) and/or the **Google Photos Picker API** (for importing from a user's Google Photos library)
3. Configure the OAuth consent screen (External, add yourself as a test user)
4. Create OAuth credentials (Desktop app type), download as `credentials.json`, place in the project root
5. On first run, `auth.py` will open a browser window for you to authorize the app; this generates `token.json`

**Note on Google Photos:** As of April 2025, Google restricted the Photos Library API — apps can no longer list a user's entire existing library without a formal Google verification process. This project uses the **Photos Picker API** instead: the user explicitly selects photos through Google's own picker UI each time, and the app receives access only to those selected items. See `ARCHITECTURE.md` for details.

### 4. Run the API

```bash
python -m uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation (Swagger UI).

### 5. Run with Docker (full stack)

```bash
docker compose up -d
```

This builds and runs both the API and database containers together. Your project folder is bind-mounted into the API container, so local storage and any bulk-downloaded datasets remain accessible.

## API Usage

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and ingest a photo (multipart file upload) |
| GET | `/photos` | List photos (paginated, limit 50) |
| GET | `/photos/{id}` | Get photo metadata |
| GET | `/photos/{id}/image` | View/download the actual image file |
| DELETE | `/photos/{id}` | Delete a photo |
| GET | `/search?q=...&top_k=5` | Natural language search |
| POST | `/dedup` | Run duplicate detection (resumable, progress-tracked) |
| GET | `/duplicates` | List detected duplicate pairs, with viewable image URLs |
| POST | `/faces/cluster` | Run face detection/clustering (resumable, progress-tracked) |
| GET | `/faces/clusters` | List face clusters, with viewable image URLs |
| POST | `/photos/{id}/categorize` | Categorize a photo (zero-shot CLIP) |
| GET | `/photos/{id}/categories` | Get a photo's assigned categories |
| GET | `/jobs/status` | Poll live progress of `/dedup` or `/faces/cluster` while running |
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

## Known Limitations

- **Face detection** uses the default HOG model, which is less reliable on angled/profile faces. CNN model (`model="cnn"`) available as a slower, more accurate alternative — not yet wired up as a config option.
- **No GPU acceleration** — CLIP and face embedding inference run on CPU.
- **Google Drive backend** has no automated retry/error handling beyond what the Drive API raises natively.
- **Google Photos** integration only provides access to photos the user actively selects in each picker session — there's no way to browse/search a user's full library from within the app before they've picked (a platform-level restriction, not a limitation of this implementation — see `ARCHITECTURE.md`).
- **Face clustering accuracy degrades on large libraries.** The incremental clustering approach (matching each new face against only its single nearest known face, rather than re-running full batch clustering) is vulnerable to error chaining: if two visually similar but distinct people are ever matched together early in a run, subsequent faces that are loosely close to either person can keep joining the same cluster, producing an oversized, incorrect "supercluster" over a long run. This was observed directly when testing against the full 13,233-image LFW dataset — one cluster grew to 1,056 faces (verified visually to include clearly different people).
  - **Why this trade-off was made:** batch DBSCAN re-clustering is accurate but doesn't scale — it would require re-processing the entire face history on every call. The incremental approach trades some accuracy for the ability to only process new photos, necessary at real library scale.
  - **How this would be fixed in production:** cap cluster size and periodically re-validate/split oversized clusters against their centroid, use a stricter tolerance combined with a minimum-confidence threshold before merging, or run a periodic batch DBSCAN reconciliation pass in the background.

## Scaling Considerations (tested up to 113,268 images)

The system was stress-tested well beyond toy-scale demos, combining the full LFW dataset (13,233 images), STL-10's unlabeled split (100,000 images), and a small Google Photos import (12 images) — **113,268 total images**, clearing the assignment's 100,000-image requirement with real, tested data.

### What scaled well

- **Bulk ingestion** (`scripts/bulk_ingest.py`) — batches 32 images per CLIP forward pass. Ingested 13,233 images in ~458 seconds, and 100,000 images in ~3,876 seconds (~26 images/sec), comfortably CPU-bound and demo-viable.
- **Search** (`/search`) — backed by a pgvector `ivfflat` index on `photos.embedding`, so queries return in milliseconds regardless of library size.
- **Face clustering re-runs** — a `faces_processed` flag per photo ensures only new/unprocessed photos are touched on subsequent calls.

### What required rework at this scale

Both `/dedup` and `/faces/cluster` were initially implemented as long-running, single-transaction endpoints that only committed once at the very end — discovered to be a real problem at 100k+ scale (long waits with no visibility, and any interruption meant losing all progress). This was fixed with:

1. **Periodic commits** — every 2,000 photos for `/dedup`, every 500 for `/faces/cluster` — instead of one commit at the end
2. **Resume flags** — a `dedup_processed` flag (mirroring the existing `faces_processed` pattern) so a restart skips already-completed photos instead of re-scanning the whole library
3. **A `job_status` table + `GET /jobs/status` endpoint** — written on each periodic commit, pollable from a separate request while the job is still running, so progress is visible in real time instead of a silent multi-hour wait
4. **Reduced query count in `/dedup`** — the original version issued an extra existence-check database query per potential duplicate match found; this was replaced with an in-memory lookup set built once at the start, cutting total query volume significantly

This was discovered through direct experience: the initial `/dedup` and `/faces/cluster` runs against the full 113k library were interrupted (laptop sleep, accidental terminal close) partway through, which under the original one-commit-at-the-end design would have meant restarting from zero each time. The fixes above make both operations genuinely production-viable at this scale rather than a fragile, unmonitorable batch job.

## Roadmap

- [ ] Background job queue (Celery + Redis) so `/dedup` and `/faces/cluster` don't block the request thread at all
- [ ] CNN face detection toggle for improved accuracy
- [ ] Periodic batch reconciliation pass to correct face clustering drift
- [ ] Web UI for browsing/search (currently Swagger UI serves as the interactive interface)
- [ ] Postman collection
