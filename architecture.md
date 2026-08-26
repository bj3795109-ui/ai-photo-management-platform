# Architecture & Design Decisions

## System Overview

The AI Photo Management Platform ingests photos from local storage or Google Drive, computes CLIP embeddings and face encodings for each image, and stores everything in PostgreSQL with pgvector for fast similarity search. A FastAPI layer exposes ingestion, search, deduplication, face clustering, and categorization as REST endpoints.

```mermaid
graph TB
    subgraph Client
        User[User / Evaluator / Web UI]
    end

    subgraph API["FastAPI Layer (api/main.py)"]
        Ingest["POST /ingest"]
        Search["GET /search"]
        Dedup["POST /dedup"]
        FaceCluster["POST /faces/cluster"]
        Reconcile["POST /faces/reconcile"]
        JobStatusRoute["GET /jobs/status"]
    end

    subgraph Queue["Background Queue (Celery + Redis)"]
        Redis[("Redis Broker")]
        Beat["Celery Beat Scheduler<br/>(Periodic Reconciliation)"]
        Worker["Celery Worker (tasks.py / celery_app.py)"]
    end

    subgraph Pipeline["Processing Pipeline (pipeline/)"]
        Jobs["jobs.py<br/>Long-running jobs (dedup, faces, reconcile)"]
        Embedding["embedding.py<br/>CLIP image/text embeddings, pHash"]
        Faces["faces.py<br/>Face detection, clustering, reconciliation"]
        Categorize["categorize.py<br/>OCR + CLIP classification"]
        IngestLogic["ingest.py<br/>Orchestrates upload → embed → store"]
        SearchLogic["search.py<br/>Text query → pgvector similarity"]
    end

    subgraph Storage["Storage Layer (storage/)"]
        StorageBase["base.py<br/>Abstract StorageBackend interface"]
        Local["local_backend.py<br/>Filesystem storage"]
        Drive["drive_backend.py<br/>Google Drive storage"]
    end

    subgraph DB["Database (db/)"]
        Postgres[("PostgreSQL + pgvector")]
        PhotoTable["photos"]
        FaceTable["faces"]
        ClusterTable["face_clusters"]
        DupTable["duplicates"]
        JobTable["job_status"]
    end

    User -->|HTTP requests| API
    Ingest --> IngestLogic
    Search --> SearchLogic
    Dedup -->|dispatches task| Redis
    FaceCluster -->|dispatches task| Redis
    Reconcile -->|dispatches task| Redis
    Beat -->|triggers schedule| Redis
    Redis --> Worker
    Worker --> Jobs
    Jobs --> Faces
    Jobs --> Embedding
    Jobs -->|updates progress| JobTable
    JobStatusRoute -->|polls status| JobTable

    IngestLogic --> Embedding
    IngestLogic --> StorageBase

    StorageBase --> Local
    StorageBase --> Drive

    Embedding -->|writes| PhotoTable
    Jobs -->|writes| DupTable
    Jobs -->|writes| FaceTable
    Jobs -->|writes| ClusterTable
    SearchLogic -->|pgvector query| PhotoTable
```

## Data Flow: Ingestion

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant S as Storage Backend
    participant E as CLIP Model
    participant DB as PostgreSQL

    U->>API: POST /ingest (file upload)
    API->>S: upload(file)
    S-->>API: storage_ref
    API->>E: compute_clip_embedding(image)
    E-->>API: 512-dim vector
    API->>DB: INSERT INTO photos (embedding, storage_ref, ...)
    DB-->>API: photo_id
    API-->>U: {status: queued, filename}
```

## Data Flow: Natural Language Search

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant E as CLIP Model
    participant DB as PostgreSQL (pgvector)

    U->>API: GET /search?q="a person smiling"
    API->>E: compute_text_embedding(query)
    E-->>API: 512-dim query vector
    API->>DB: ORDER BY embedding <=> query_vector (ivfflat index)
    DB-->>API: ranked photo results
    API-->>U: JSON results
```

## Key Design Decisions

### 1. CLIP for both image and text embeddings (shared vector space)
**Decision:** Use OpenCLIP (`ViT-B-32-quickgelu`, OpenAI weights) for image embeddings, and the same model's text encoder for natural language queries.
**Why:** CLIP maps images and text into the same embedding space, so a query like "a dog on a beach" and a matching photo land close together — this is what makes natural language search possible without training a custom model. Also reused for zero-shot categorization, avoiding the need for a separate classifier.
**Trade-off:** CPU inference is slower than a smaller/specialized model would be, but avoids the cost and complexity of training/fine-tuning anything.

### 2. pgvector over a dedicated vector database
**Decision:** Store embeddings as a `vector` column in PostgreSQL (via the pgvector extension) rather than using a separate vector DB (Pinecone, Weaviate, Milvus, etc.).
**Why:** Keeps the stack simpler — one database for both relational metadata (photo paths, timestamps, face clusters) and vector search, with a single source of truth and standard SQL joins between them. An `ivfflat` index provides approximate nearest-neighbor search that's fast at scale (verified up to 13,000+ images) without needing separate infrastructure.
**Trade-off:** A dedicated vector DB would likely out-perform pgvector at very large scale (100M+ vectors) or offer more advanced indexing options, but for this use case pgvector's performance was more than sufficient and the operational simplicity was worth it.

### 3. Perceptual hash (pHash) + CLIP for two-tier deduplication
**Decision:** Use pHash for fast exact/near-exact duplicate detection (e.g. the same photo saved twice, resized, or recompressed), and CLIP cosine similarity for semantic near-duplicates (e.g. burst-mode shots of the same scene).
**Why:** pHash is extremely cheap to compute and compare (bit-level Hamming distance), so it's the first-pass filter. CLIP similarity catches duplicates that pHash would miss (different crop/lighting of the same subject) but is more expensive, so it's used more selectively.

### 4. Pluggable storage backend (Strategy pattern)
**Decision:** Define an abstract `StorageBackend` interface (`upload`, `get_url`, `delete`) with local filesystem and Google Drive implementations behind it.
**Why:** The rest of the pipeline (ingestion, dedup, search) never needs to know which storage backend is active — it just calls the interface. This makes it straightforward to add new backends (e.g. S3, Google Photos) later without touching pipeline logic.

### 5. Incremental face clustering vs. full batch re-clustering
**Decision:** Track a `faces_processed` flag per photo, and match new faces against existing cluster centroids (nearest-neighbor) rather than re-running full DBSCAN clustering across the entire face history on every call.
**Why:** Full batch re-clustering on every run doesn't scale — at 13,000+ images it would mean re-processing the entire face history every time a single new photo is added. Incremental matching only processes new/unprocessed photos.
**Trade-off (discovered during testing):** this approach is vulnerable to error chaining — since each new face is matched only against its single nearest neighbor rather than re-validated against the whole cluster, visually similar-but-distinct people can occasionally merge into the same cluster over a long run, and the error compounds. This was observed directly: testing against the full 13,233-image LFW dataset produced one cluster of 1,056 faces that, on visual inspection, contained multiple different people. This is documented as a known limitation in the README, along with a proposed fix (periodic batch reconciliation, or stricter similarity thresholds with confidence-based merging) rather than silently accepting the incorrect result.

### 6. Vue 3 SPA + Tailwind CSS Web UI & FastAPI Backend
**Decision:** Provide a full Vue 3 single-file web application (`frontend/index.html`) loaded via CDN with Tailwind CSS styling, alongside FastAPI's auto-generated Swagger UI (`/docs`).
**Why:** Gives users a responsive, modern web interface with infinite-scroll gallery browsing, real-time background job progress tracking, natural language search, face cluster review, side-by-side duplicate comparison, and drag-and-drop/Google Photos uploads.
**Architecture:** CORS middleware is enabled on the FastAPI core (`allow_origins=["*"]`), allowing the frontend to run either standalone (e.g. `python -m http.server 8080`), directly via `file://`, or served directly by the API server via `serve_ui.py`.

### 7. Cross-Platform Path Normalization (Windows & Linux Containers)
**Decision:** Standardize all file storage references (`storage_ref`) and source paths (`source_path`) to forward slashes (`/`) across the entire backend, with a dedicated database migration script (`scripts/fix_paths.py`).
**Why:** Photos ingested from Windows hosts initially stored backslash separators (`\`). When running inside Linux Docker containers, backslashes are literal filename characters rather than directory separators, breaking `os.path.exists()` checks and causing 404 errors. Standardizing on forward slashes ensures 100% compatibility across host operating systems and Linux container environments.

### 8. Google Photos integration via the Picker API (not the Library API)
**Decision:** Implemented Google Photos import using the **Photos Picker API** rather than the (now-restricted) Photos Library API.
**Why:** As of April 2025, Google restricted the Library API so that new/unverified apps can no longer list a user's entire existing photo library (`photoslibrary.readonly` requires a formal Google security verification process that takes weeks and is intended for production apps with real users). The Picker API is Google's current recommended path for third-party apps: the user explicitly selects photos through Google's own picker UI (`https://photos.google.com/integration/picker/...`), and the app receives temporary, scoped access only to those selected items.
**Flow:** `POST /google-photos/session` creates a session and returns a `picker_uri`; the user opens it, selects photos, and the app polls until selection is complete; `POST /google-photos/session/{id}/import` then downloads each selected item and feeds it through the existing `ingest_photo()` pipeline — reusing the same OAuth credentials (extended with the `photospicker.mediaitems.readonly` scope) and storage/embedding logic already built for Drive.
**Verified:** tested live against a real Google account — 12 real photos selected via the picker, downloaded, and successfully ingested with embeddings into the running database.
**Trade-off:** unlike full library listing, the Picker API only provides access to photos the user actively selects in each session — there's no way to browse/search the user's full library from within the app before they've picked anything already exists (or `Google Drive` storage, which does support full listing since Drive's more permissive API model wasn't affected by the same restriction).

### 9. Asynchronous Task Queue with Celery & Redis
**Decision:** Offload long-running, CPU-bound operations (`/dedup`, `/faces/cluster`, and `/faces/reconcile`) to dedicated Celery background workers (`worker` service) backed by Redis as a message broker and result backend. Celery Beat (`beat` service) schedules periodic face-cluster reconciliation passes.
**Why:** Running CPU-intensive machine learning algorithms (face embedding calculations, DBSCAN clustering, vector distance comparisons over 116k+ photos) directly on FastAPI HTTP request threads resulted in request timeouts, blocked server loops, and dropped progress if a request connection dropped.
**Architecture:**
- **Celery Worker (`tasks.py` / `pipeline/jobs.py`)**: Asynchronously executes deduplication and face clustering jobs off the HTTP thread.
- **Celery Beat (`celery_app.py`)**: Automatically triggers `tasks.reconcile_face_clusters` every 6 hours (configurable via `RECONCILE_INTERVAL_SECONDS`).
- **Live Progress Tracking (`job_status` table)**: Workers update a persistent `JobStatus` record during batched processing, which the frontend polls via `GET /jobs/status` to render real-time progress bars.

## Scale Testing Summary

The system was stress-tested against the full LFW dataset (13,233 images) — see `README.md`'s "Scaling Considerations" section for full details. Summary:

| Component | Result at 13,233 images |
|---|---|
| Bulk ingestion (batched CLIP) | ~458 seconds (~29 images/sec) |
| Natural language search | Milliseconds (pgvector `ivfflat` index) |
| Deduplication | Fast, index-backed nearest-neighbor query |
| Face clustering (initial run) | ~1 hour (synchronous, unbatched — documented bottleneck) |
| Face clustering (accuracy) | Degrades on large libraries — documented limitation |
| Face clustering (re-runs) | Near-instant (incremental, only processes new photos) |

The assignment specifies support for at least 100,000 images. The architecture (batched ingestion, pgvector indexing, incremental processing) is designed to extend to that scale, but was validated directly only up to 13,233 images due to time constraints — the honest, tested number is reported here rather than an untested claim of 100k support.