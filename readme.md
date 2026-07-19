# AI Photo Management Platform

An AI-powered photo library manager that deduplicates, organizes by face, and enables natural-language search over your personal photo collection.

Built as part of a PanScience Innovations internship assignment.

## Features

- **Perceptual + semantic deduplication** — detects near-identical and visually similar photos using pHash and CLIP embeddings
- **Face clustering** — groups photos by the people in them using `face_recognition` + DBSCAN, with incremental matching for new uploads
- **Zero-shot categorization** — tags photos by scene/content (people, food, documents, nature, etc.) using CLIP, no training required
- **Natural language search** — find photos by describing them ("a person smiling", "sunset at the beach") via CLIP text-image similarity, powered by pgvector
- **Pluggable storage** — local filesystem backend implemented; Google Drive backend implemented and OAuth-tested (5TB storage via Gmail)

## Tech Stack

- **Embeddings**: OpenCLIP (`ViT-B-32-quickgelu`, OpenAI weights)
- **Face detection/clustering**: `face_recognition` (dlib) + scikit-learn DBSCAN
- **Database**: PostgreSQL + pgvector (via Docker Compose)
- **ORM**: SQLAlchemy
- **API**: FastAPI
- **Storage**: Local filesystem / Google Drive (OAuth 2.0)
- **Testing**: pytest

## Architecture

```
pipeline/
  embedding.py     # pHash, CLIP image/text embeddings
  dedup.py          # pHash + CLIP-based duplicate detection
  faces.py           # Face detection, clustering, incremental matching
  categorize.py       # Zero-shot scene/content tagging
  ingest.py             # End-to-end: upload -> embed -> store
  search.py               # Natural language search via pgvector

storage/
  base.py          # Abstract StorageBackend interface
  local_backend.py  # Local filesystem implementation
  drive_backend.py    # Google Drive implementation

db/
  database.py      # SQLAlchemy engine/session setup
  models.py          # Photo, Face, FaceCluster, Duplicate, Category tables

tests/
  test_*.py         # Unit tests for each pipeline module
  fixtures.py          # Synthetic test image generator
```

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt --break-system-packages
```

`dlib` requires a prebuilt wheel on Windows (avoids needing Visual Studio build tools):
```bash
pip install dlib-bin
pip install face_recognition --no-deps
```

### 2. Database

```bash
docker compose up -d db
```

Runs Postgres with the pgvector extension pre-installed (`ankane/pgvector` image).

Create tables:
```python
from db.database import engine, Base
from db import models
Base.metadata.create_all(bind=engine)
```

### 3. Google Drive backend (optional)

1. Create a Google Cloud project, enable the Drive API
2. Configure OAuth consent screen (External, add yourself as a test user)
3. Create an OAuth Client ID (Desktop app), download as `credentials.json` into the project root
4. First run will open a browser for one-time authorization, then cache a reusable `token.json`

## Usage

```python
from pipeline.ingest import ingest_photo
from pipeline.search import search_photos

# Ingest a photo (uploads to storage, computes embedding, stores in DB)
photo_id = ingest_photo("path/to/photo.jpg", "photo.jpg")

# Search naturally
results = search_photos("a person smiling", top_k=5)
```

## Testing

```bash
python -m pytest -v
```

15 tests covering embeddings, deduplication, face clustering/matching, categorization, and storage. Synthetic fixture images are generated via `tests/fixtures.py` (no real photos required to run tests).

**Known gap**: `detect_faces` is not directly unit tested since it requires a real photo containing a face; `cluster_faces` and `match_face_to_cluster` are tested in isolation using synthetic 128-dim encodings.

## Known Limitations

- Face detection uses the default HOG model, which is less reliable on angled/profile faces. CNN model (`model="cnn"`) available as a slower, more accurate alternative — not yet wired up as a config option.
- No GPU acceleration used; CLIP and face embedding inference run on CPU.
- Google Drive backend has no automated retry/error handling beyond what the Drive API raises natively.
- **Face clustering accuracy degrades on large libraries.** The incremental clustering approach (matching each new face against only its single nearest known face, rather than re-running full batch clustering) is vulnerable to error chaining: if two visually similar but distinct people are ever matched together early in a run, subsequent faces that are loosely close to either person can keep joining the same cluster, producing an oversized, incorrect "supercluster" over a long run. This was observed directly when testing against the full 13,233-image LFW dataset — one cluster grew to 1,056 faces (verified visually to include clearly different people), while the true `cluster_faces` (batch DBSCAN) function, tested in isolation via unit tests, does not exhibit this failure mode since it considers local density across the whole batch rather than nearest-single-point matching.
  - **Why this trade-off was made:** batch DBSCAN re-clustering is accurate but doesn't scale — it would require re-processing the entire face history on every call (see "Scaling Considerations" below). The incremental approach trades some accuracy for the ability to only process new photos, which is necessary at real library scale.
  - **How this would be fixed in a production system:** cap cluster size and re-validate/split oversized clusters against their centroid periodically, use a stricter tolerance combined with a minimum-confidence threshold before merging, or run a periodic batch DBSCAN reconciliation pass in the background to correct drift from the incremental process without blocking real-time ingestion.

## Scaling Considerations (tested up to 13,000+ images)

The system was stress-tested against the full LFW dataset (13,233 images) to validate behavior beyond toy-scale demos. Findings:

**What scaled well:**
- **Bulk ingestion** (`scripts/bulk_ingest.py`) — batches 32 images per CLIP forward pass instead of one at a time. Ingested all 13,233 images in ~458 seconds (~29 images/sec), comfortably CPU-bound and demo-viable.
- **Search** (`/search`) — backed by a pgvector `ivfflat` index on `photos.embedding` (cosine distance), so queries return in milliseconds regardless of library size, instead of an O(n) or O(n²) Python-side scan.
- **Deduplication** (`/dedup`) — rewritten to use the same pgvector index (per-photo nearest-neighbor query) rather than pairwise comparison in Python, avoiding an O(n²) blowup at scale.
- **Face clustering re-runs** — a `faces_processed` flag on `Photo` ensures only new/unprocessed photos are touched on subsequent calls, avoiding a full re-cluster of the entire library every time.

**What did not scale well (known bottleneck):**
- **Initial face detection/clustering** (`/faces/cluster`) is synchronous and unbatched — each image runs through `face_recognition`'s HOG detector one at a time, inside a single blocking HTTP request. On 13,000+ images this took on the order of an hour on a modern CPU (Intel Core Ultra 5 225H), with no progress visibility during the run and a real risk of client-side timeouts.

**Root cause:** unlike CLIP inference (which is naturally batchable via PyTorch tensor ops), `face_recognition`'s HOG detector operates on one image at a time and offers no batching, and the endpoint has no background task or progress reporting — it computes everything before returning a single response.

**How this would be fixed in a production system:**
1. Move face detection off the request path entirely — into a background job (e.g. Celery + Redis, or even FastAPI `BackgroundTasks` for a lighter-weight version) so `/faces/cluster` returns immediately with a job ID, and the client polls a status endpoint instead of blocking on the HTTP connection.
2. Add progress logging/checkpointing (commit incrementally per batch rather than once at the end) so a long-running job can be interrupted and resumed without losing completed work.
3. Evaluate the CNN face detector's batching support, or an alternative face detection library with native batch inference, if throughput becomes the primary constraint over accuracy.

This gap was deliberately left as-is (rather than "fixed" superficially) to give an honest picture of where the current architecture holds up under real load and where it doesn't — the ingestion and search paths were prioritized since they're on the user-facing hot path, while one-time bulk face clustering is a background/maintenance operation that can tolerate being slow, just not silent.

## Roadmap

- [ ] Background job queue (Celery + Redis) for face clustering and other long-running batch operations
- [ ] CNN face detection toggle for improved accuracy
- [ ] Google Drive ingestion source (currently storage-only)
- [ ] Web UI for browsing/search
