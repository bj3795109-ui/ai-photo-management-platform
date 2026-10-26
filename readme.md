# AI Photo Management Platform

An AI-powered photo management platform that connects to Google Photos and local storage, identifies duplicate and near-duplicate images, categorizes photos, groups people using facial recognition, and supports natural-language search.

Built with **FastAPI, PostgreSQL + pgvector, OpenCLIP, Tesseract OCR, Redis, and Celery**, with a Vue-based web interface.

**Scale tested:** 116,000+ images across multiple datasets and import sources.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system architecture diagrams, design decisions, processing flows, and scalability considerations.

## Features

* **Web UI** — Gallery with infinite scroll, natural-language search, face clusters, duplicate review, and upload/import from local devices, Google Photos, and Google Drive.
* **Image ingestion** — Import images from local storage and Google Photos through the Picker API.
* **Natural-language search** — Search your photo library using CLIP image-text embeddings and pgvector similarity queries.
* **Duplicate detection** — Detect exact/near-exact duplicates using perceptual hashing and semantically similar images using CLIP.
* **Face grouping** — Detect and group faces across a photo collection using face embeddings and clustering.
* **Two-path categorization**:

  * Text-heavy images such as documents, prescriptions, and receipts are classified using OCR and keyword matching.
  * Visual images such as people, pets, travel, food, nature, vehicles, and screenshots are classified using CLIP zero-shot classification.
* **Bulk ingestion** — Batched CLIP inference for processing large image libraries.
* **Background job queue** — Deduplication, face clustering, and reconciliation run asynchronously through Celery workers and Redis.
* **Persistent job tracking** — Monitor processing progress and resume supported operations after interruption.
* **Periodic face-cluster reconciliation** — Scheduled DBSCAN-based processing to identify and split inconsistent face clusters.
* **Docker Compose** — Containerized infrastructure for reproducible local execution.

## Tech Stack

| Component                       | Technology                                       |
| ------------------------------- | ------------------------------------------------ |
| Backend API                     | FastAPI                                          |
| Frontend                        | Vue 3 (CDN-loaded) + Tailwind CSS                |
| Database                        | PostgreSQL                                       |
| Vector Search                   | pgvector                                         |
| Background Processing           | Celery                                           |
| Message Broker / Result Backend | Redis                                            |
| Task Scheduling                 | Celery Beat                                      |
| Image and Text Embeddings       | OpenCLIP (`ViT-B-32-quickgelu`, OpenAI weights)  |
| Document Classification         | Tesseract OCR (`pytesseract`) + keyword matching |
| Face Detection                  | `face_recognition` (HOG/CNN)                     |
| Face Clustering                 | Incremental matching + DBSCAN reconciliation     |
| Duplicate Detection             | pHash + CLIP similarity                          |
| Storage                         | Local filesystem / Google Drive                  |
| Photo Import                    | Google Photos Picker API                         |
| Containerization                | Docker + Docker Compose                          |

---

## Setup

### Prerequisites

* Python 3.11+ (Python 3.13 tested)
* Docker Desktop
* Tesseract OCR
* Google Cloud project (optional, for Google Drive storage and Google Photos import)

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/bj3795109-ui/ai-photo-management-platform.git
cd ai-photo-management-platform

python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

**Important (Windows):**

The `face_recognition` package depends on `dlib`, which may fail to build without Visual Studio C++ Build Tools.

This project uses `dlib-bin`, a prebuilt wheel, instead.

After installing the main dependencies, install `face_recognition` without pulling its `dlib` dependency:

```bash
pip install face_recognition --no-deps
```

### 2. Install Tesseract OCR

The `pytesseract` Python package is a wrapper around the Tesseract OCR executable.

The executable must be installed separately.

**Windows:** Install Tesseract from the [UB Mannheim distribution](https://github.com/UB-Mannheim/tesseract/wiki).

The default installation path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The application already checks this location.

**macOS:**

```bash
brew install tesseract
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

**Docker:**

Tesseract is installed through the project's Dockerfile.

Verify the installation:

```python
import pytesseract

print(pytesseract.get_tesseract_version())
```

### 3. Start PostgreSQL and Redis

```bash
docker compose up -d db redis
```

This starts the PostgreSQL database with pgvector and the Redis service required by Celery.

### 4. Google Cloud Setup (Optional)

Google Cloud credentials are required for Google Drive storage and Google Photos imports.

1. Create a project in [Google Cloud Console](https://console.cloud.google.com).
2. Enable the Google Drive API and/or Google Photos Picker API.
3. Configure the OAuth consent screen.
4. Add your account as a test user.
5. Create OAuth credentials using the Desktop application type.
6. Download the credentials as `credentials.json`.
7. Place the file in the project root.
8. Complete authorization when prompted by `auth.py`.

Successful authorization generates `token.json`.

**Google Photos API restrictions:**

Google restricted third-party access to existing Google Photos libraries beginning in April 2025.

This project uses the Google Photos Picker API, which allows users to explicitly select photos for import.

The application can access the selected items rather than automatically enumerating the user's entire library.

**OAuth token expiration:**

Google OAuth applications in Testing mode may issue refresh tokens that expire after seven days.

Periodic reauthorization may therefore be necessary.

### 5. Run the API, Worker, and Scheduler

Start FastAPI:

```bash
python -m uvicorn api.main:app --reload
```

Start the Celery worker in a separate terminal:

```bash
celery -A celery_app worker --loglevel=info
```

Start Celery Beat in another terminal:

```bash
celery -A celery_app beat --loglevel=info
```

The worker executes background deduplication, face clustering, and reconciliation jobs.

Celery Beat schedules periodic face-cluster reconciliation.

**Note:** The API can start without the worker and scheduler, but queued tasks will not execute until a worker is available.

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The frontend can be opened directly from:

```text
frontend/index.html
```

Alternatively, serve it using:

```bash
cd frontend
python -m http.server 8080
```

#### Existing Installation / Path Migration

When migrating stored data from Windows to Linux or Docker, normalize previously stored file paths:

```bash
python -m scripts.fix_paths
```

#### Google OAuth Reauthentication

If an OAuth refresh token expires, run:

```bash
python scripts/reauth.py
```

The API also attempts reauthentication during startup when necessary.

### 6. Run the Full Stack with Docker

```bash
docker compose up -d
```

Docker Compose starts five services:

| Service  | Responsibility                   |
| -------- | -------------------------------- |
| `db`     | PostgreSQL + pgvector            |
| `redis`  | Celery broker and result backend |
| `api`    | FastAPI backend                  |
| `worker` | Background job execution         |
| `beat`   | Periodic task scheduling         |

Check running services:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

Stop the stack:

```bash
docker compose down
```

The project directory is bind-mounted into the API and worker containers so that local image storage and downloaded datasets remain accessible.

---

## Web UI

The frontend is implemented using Vue 3 and Tailwind CSS.

Open:

```text
frontend/index.html
```

with FastAPI running at:

```text
http://127.0.0.1:8000
```

The interface provides:

* **Gallery** — Infinite-scroll browsing of ingested images.
* **Search** — Natural-language image search.
* **Upload & Import** — Local uploads, Google Drive, and Google Photos Picker integration.
* **Faces** — Browse detected face clusters.
* **Duplicates** — Review duplicate image pairs side by side.
* **Dashboard** — Library statistics and background job progress.

The Google Photos import interface tracks states such as:

```text
idle → picking → importing → done/error
```

---

## API Usage

| Method | Endpoint                             | Description                                  |
| ------ | ------------------------------------ | -------------------------------------------- |
| GET    | `/health`                            | Health check                                 |
| POST   | `/ingest`                            | Upload and ingest an image                   |
| GET    | `/photos?offset=0&limit=50`          | Paginated image listing                      |
| GET    | `/photos/{id}`                       | Retrieve image metadata                      |
| GET    | `/photos/{id}/image`                 | Retrieve image content                       |
| DELETE | `/photos/{id}`                       | Delete an image                              |
| GET    | `/search?q=...&top_k=5`              | Natural-language image search                |
| POST   | `/dedup`                             | Queue duplicate detection                    |
| GET    | `/duplicates?offset=0&limit=50`      | Retrieve duplicate pairs                     |
| POST   | `/faces/cluster?model=hog\|cnn`      | Queue face detection and clustering          |
| POST   | `/faces/reconcile`                   | Trigger face-cluster reconciliation          |
| GET    | `/faces/clusters`                    | Retrieve face clusters                       |
| POST   | `/photos/{id}/categorize`            | Categorize an image                          |
| GET    | `/photos/{id}/categories`            | Retrieve assigned categories                 |
| GET    | `/jobs/status`                       | Retrieve background job progress by job name |
| GET    | `/jobs/{task_id}`                    | Retrieve status of a specific task           |
| POST   | `/google-photos/session`             | Start a Google Photos Picker session         |
| POST   | `/google-photos/session/{id}/import` | Import selected Google Photos images         |

### Background Task Responses

The following endpoints enqueue tasks instead of executing expensive processing inside the HTTP request:

```http
POST /dedup
POST /faces/cluster
POST /faces/reconcile
```

They return a task identifier that can be used to monitor progress:

```http
GET /jobs/{task_id}
```

Persistent job progress is also available through:

```http
GET /jobs/status
```

---

## Bulk Ingestion

For processing large image collections, use the bulk-ingestion script:

```bash
python -m scripts.bulk_ingest /path/to/folder
```

The script uses batched CLIP inference rather than performing a separate model forward pass for every image.

The documented batch size is **32 images**.

This improves CPU inference throughput during large-scale ingestion.

Bulk ingestion is distinct from downstream operations such as face clustering, duplicate detection, and OCR classification.

Completing ingestion does not necessarily mean those operations have also completed.

---

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

The repository includes 15+ tests covering:

* Embedding generation
* Duplicate detection
* Face detection and clustering
* Image categorization
* Storage operations

---

## Categorization: Why Two Classification Paths?

Initial testing revealed that CLIP-only categorization was unreliable for text-based categories such as:

* Documents
* Prescriptions
* Receipts

For example, searching for *prescription* returned images of people wearing eyeglasses.

CLIP associated the term with prescription glasses rather than reliably identifying prescription documents.

This exposed a limitation of relying exclusively on visual-semantic similarity for document classification.

### Solution: OCR + CLIP

The platform uses two classification paths.

**Path 1: Text-heavy images**

`pipeline/document_classifier.py` runs Tesseract OCR on an image.

If enough text is detected, keyword matching determines the document category.

Examples:

| Category      | Example Keywords                                            |
| ------------- | ----------------------------------------------------------- |
| Prescriptions | `rx`, `dosage`, `pharmacy`                                  |
| Receipts      | `total`, `subtotal`, `tax`                                  |
| Documents     | Text-heavy images without a more specific matching category |

**Path 2: Visual images**

If the image does not meet the minimum OCR word-count threshold, classification falls back to CLIP zero-shot classification.

Supported visual categories include:

* People
* Pets
* Travel
* Food
* Nature
* Vehicles
* Screenshots
* Other

### Validation

The OCR-based approach was tested using the **Tobacco3482 dataset**, which contains scanned business and legal documents.

After introducing OCR, tested document images were classified using extracted text rather than relying exclusively on CLIP's visual predictions.

### Trade-offs

OCR and keyword matching are transparent and relatively easy to debug.

However, the approach depends on OCR quality and predefined vocabulary.

Poor image quality, unfamiliar document formats, and missing keywords can reduce classification accuracy.

---

## Background Jobs (Celery + Redis)

The original implementation executed expensive operations directly inside FastAPI request handlers.

This became problematic when processing larger image collections.

Operations such as duplicate detection and face clustering could take considerable time, leaving HTTP requests open and increasing the risk of timeouts.

### Current Architecture

The following endpoints now dispatch tasks to Celery:

```http
POST /dedup
POST /faces/cluster
POST /faces/reconcile
```

**FastAPI** receives requests and enqueues tasks.

**Redis** transports queued tasks to workers.

**Celery workers** execute the CPU-intensive operations.

**Celery Beat** schedules periodic reconciliation.

**PostgreSQL** stores persistent processing state and job progress.

### Benefits

* Expensive operations no longer occupy HTTP request handlers for their entire execution.
* Users can monitor job progress through separate requests.
* Processing state can be persisted between batches.
* Supported operations can skip previously processed images after interruption.
* Periodic reconciliation can execute without manual intervention.

### Job Monitoring

Poll job status by operation name:

```http
GET /jobs/status
```

Or by Celery task identifier:

```http
GET /jobs/{task_id}
```

Workers write progress to the same PostgreSQL-backed `job_status` table used by the API.

### Configuration

Redis connection:

```text
REDIS_URL=redis://localhost:6379/0
```

Periodic reconciliation interval:

```text
RECONCILE_INTERVAL_SECONDS=21600
```

The default interval is six hours.

---

## Face-Cluster Reconciliation

Incremental face clustering allows the platform to process newly detected faces without repeatedly clustering the entire face collection.

The incremental implementation matches each new face against its nearest known face.

This reduces repeated computation but introduces a potential accuracy problem.

### Problem: Cluster Drift

If two visually similar but distinct individuals are incorrectly grouped, subsequent matches can reinforce the incorrect grouping.

Over time, a cluster may contain faces belonging to multiple people.

This issue was observed during testing on the **13,233-image LFW dataset**.

One cluster grew to **1,056 faces** and was visually verified to contain multiple distinct individuals.

### Implemented Solution

The platform implements periodic batch reconciliation using DBSCAN.

The reconciliation logic is implemented in:

```text
pipeline/jobs.py::run_reconciliation_job()
```

It is connected to the Celery task:

```text
tasks.reconcile_face_clusters
```

Reconciliation runs periodically through Celery Beat and can also be triggered manually:

```http
POST /faces/reconcile
```

### Reconciliation Process

1. Identify face clusters exceeding a configurable size threshold.
2. Retrieve the face encodings belonging to each eligible cluster.
3. Re-cluster those faces using DBSCAN with a stricter tolerance.
4. Compare the resulting groups with the existing cluster.
5. Preserve the original cluster ID for the largest resulting group.
6. Assign new cluster IDs to additional groups.

The batch-clustering implementation is located in:

```text
pipeline/faces.py::cluster_faces()
```

### Configuration

| Variable                     | Default | Description                               |
| ---------------------------- | ------- | ----------------------------------------- |
| `RECONCILE_SIZE_THRESHOLD`   | `30`    | Cluster-size threshold for reconciliation |
| `RECONCILE_TOLERANCE`        | `0.5`   | DBSCAN clustering tolerance               |
| `RECONCILE_INTERVAL_SECONDS` | `21600` | Scheduling interval in seconds            |

### Trade-off

Incremental matching is computationally efficient but can introduce clustering errors.

Periodic reconciliation adds CPU overhead in exchange for a mechanism to revisit and correct some inconsistent clusters.

This approach improves the system's ability to manage clustering drift, but it does not guarantee perfect face identification.

---

## Face Detection Model

The face-detection model is configured using:

```text
FACE_DETECTION_MODEL
```

The default value is:

```text
hog
```

Two options are supported.

| Model | Characteristics                                                            |
| ----- | -------------------------------------------------------------------------- |
| `hog` | Faster on CPU; less reliable for angled or profile faces                   |
| `cnn` | More computationally expensive; can improve detection of non-frontal faces |

The model can be overridden for a clustering request:

```http
POST /faces/cluster?model=cnn
```

CNN inference is significantly more expensive without GPU acceleration.

---

## Scaling Considerations

### Dataset Coverage: 116,000+ Images

The platform has processed images from multiple datasets and import sources.

| Dataset                |            Images | Primary Testing Purpose         |
| ---------------------- | ----------------: | ------------------------------- |
| LFW                    |            13,233 | Face detection and clustering   |
| STL-10 unlabeled split |           100,000 | Large-scale ingestion           |
| Tobacco3482            |             3,482 | OCR and document classification |
| Google Photos imports  | Additional images | External import validation      |

The combined workload exceeds **116,000 images**.

**Important:** This figure represents the combined image-processing workload across datasets and sources. It does not mean every feature was independently benchmarked across all 116K+ images.

### What Scaled Well

#### 1. Bulk Ingestion

The bulk-ingestion implementation processes images in batches of 32 during CLIP inference.

Recorded benchmarks:

| Dataset |  Images | Ingestion Time | Approximate Throughput |
| ------- | ------: | -------------: | ---------------------: |
| LFW     |  13,233 |    458 seconds |          29 images/sec |
| STL-10  | 100,000 |  3,876 seconds |          26 images/sec |

These measurements demonstrate large-scale bulk ingestion using CPU-based inference.

They are ingestion benchmarks, not end-to-end measurements of all processing features.

#### 2. Vector Search

Natural-language search uses CLIP text embeddings and PostgreSQL pgvector similarity queries.

An `ivfflat` index supports approximate nearest-neighbor retrieval.

This avoids relying exclusively on exhaustive vector comparisons.

However, query latency depends on dataset size, index configuration, available hardware, and database workload.

The current documentation does not claim constant search latency or verified latency percentiles at the full 116K+ image scale.

#### 3. Incremental Face Processing

A `faces_processed` flag allows subsequent clustering runs to skip previously processed images.

This reduces unnecessary repeated work when new images are added to an existing library.

#### 4. Paginated Gallery

The gallery uses pagination and infinite scrolling rather than loading the entire photo library in a single response.

This reduces the amount of image metadata transferred per request and limits the number of images rendered at once.

---

## Engineering Challenges at Scale

Testing on larger image collections exposed limitations in the original implementation.

### 1. Long-Running HTTP Requests

**Problem:**

Deduplication and face clustering originally executed directly inside FastAPI request handlers.

Large workloads could keep requests active for extended periods and risk HTTP timeouts.

**Solution:**

Move expensive processing into Celery workers using Redis.

The API now queues tasks and returns task identifiers.

### 2. Large Database Transactions

**Problem:**

The original implementation committed processing results only after completing an entire operation.

If processing was interrupted, uncommitted work could be lost.

**Solution:**

Introduce periodic database commits.

Documented commit intervals:

| Operation       |    Commit Interval |
| --------------- | -----------------: |
| Deduplication   | Every 2,000 photos |
| Face clustering |   Every 500 photos |

This limits the amount of uncommitted work during long-running jobs.

### 3. Reprocessing After Interruption

**Problem:**

Restarting an interrupted operation could require processing the same images again.

**Solution:**

Introduce persistent processing flags.

```text
dedup_processed
faces_processed
```

These flags allow supported operations to skip images that were processed previously.

### 4. Excessive Database Queries

**Problem:**

The original duplicate-detection implementation issued additional existence-check queries for potential duplicate matches.

At larger dataset sizes, these repeated database operations increased overhead.

**Solution:**

Replace repeated existence checks with an in-memory lookup set initialized at the beginning of processing.

This reduces redundant database queries.

**Trade-off:** The lookup set consumes additional application memory.

### 5. Limited Progress Visibility

**Problem:**

Long-running operations initially provided little visibility into processing progress.

**Solution:**

Introduce a persistent `job_status` table and progress endpoints.

Workers update job status during processing.

Users can retrieve progress without keeping the original HTTP request open.

### 6. Face-Cluster Drift

**Problem:**

Incremental nearest-neighbor matching occasionally grouped distinct individuals together.

The LFW test exposed a cluster containing 1,056 faces from multiple individuals.

**Solution:**

Implement periodic DBSCAN-based reconciliation using Celery Beat and Celery workers.

This provides a mechanism to revisit and split inconsistent clusters.

---

## Lessons from Large-Scale Testing

The initial deduplication and face-clustering implementations were tested against a library containing approximately 113,000 images.

During these runs, processing was interrupted by events such as laptop sleep and accidental terminal closure.

The original single-transaction design risked losing substantial progress.

These failures motivated several architectural changes:

1. Periodic database commits.
2. Persistent processing flags.
3. Resumable processing.
4. PostgreSQL-backed job progress tracking.
5. Reduced duplicate-detection query overhead.
6. Celery-based background execution.
7. Scheduled face-cluster reconciliation.

These changes improve reliability and maintainability under larger workloads.

They do not eliminate all failure modes, but they reduce the operational fragility of long-running image-processing jobs.

---

## Known Limitations

### 1. CPU-Based Inference

CLIP and face-processing operations currently run without GPU acceleration.

This limits inference throughput, particularly for CNN-based face detection.

### 2. Face-Clustering Accuracy

Incremental matching can introduce identity-grouping errors.

Periodic reconciliation provides a correction mechanism but does not guarantee perfect clustering accuracy.

### 3. OCR Quality

Tesseract OCR performance depends on image quality.

Blurry, low-resolution, rotated, or heavily skewed document images may produce inaccurate text.

This can cause document classification to fail or fall back to CLIP.

### 4. Keyword-Based Document Classification

Document classification relies on predefined keywords rather than a trained text classifier.

This makes the system transparent and debuggable but limits its ability to generalize to unfamiliar document formats.

### 5. Google Drive Error Handling

The Google Drive backend does not provide comprehensive automated retry handling beyond the underlying API behavior.

### 6. Google Photos Picker Restrictions

The Google Photos Picker API provides access only to images explicitly selected by the user.

The application cannot automatically browse an entire existing Google Photos library through the Picker API.

### 7. Database Migration Management

Database migrations currently use manual and inline scripts.

For example, the `created_at` column migration uses an idempotent SQL block during startup and a standalone migration script:

```text
scripts/add_created_at.py
```

A production deployment would benefit from versioned migrations using Alembic.

### 8. Search Performance Benchmarking

Although the platform uses indexed vector similarity search, it does not currently document repeatable latency-percentile benchmarks across the entire 116K+ image workload.

A dedicated benchmark would be needed to quantify search latency, recall, and performance under concurrent requests.

---

## Roadmap

* [ ] Introduce an ML-based document classifier to complement or replace keyword matching.
* [ ] Add Alembic for versioned database migrations.
* [ ] Provide a Postman collection.
* [ ] Dynamically adjust cluster-size thresholds and reconciliation tolerance.
* [ ] Add repeatable vector-search latency and recall benchmarks.
* [ ] Improve background-job failure handling and recovery.
* [ ] Evaluate GPU acceleration for computationally expensive inference workloads.

---

## Architecture Documentation

For additional technical details, refer to [ARCHITECTURE.md](./ARCHITECTURE.md).

It covers:

* High-level system architecture
* Image-ingestion and semantic-search flows
* Celery and Redis background processing
* Storage abstraction
* pgvector indexing decisions
* Face-clustering reconciliation
* OCR-based document classification
* Scalability testing and engineering trade-offs
