import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
from celery.result import AsyncResult
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from celery_app import celery_app
from db.database import get_db, engine, Base
from db.models import Photo, Duplicate, FaceCluster, Face, Category, JobStatus
from pipeline.ingest import ingest_photo
from pipeline.search import search_photos as search_photos_fn
from pipeline.categorize import categorize_image
from auth import get_drive_credentials
from tasks import run_dedup_task, run_face_clustering_task, reconcile_face_clusters_task
from sources.google_photos_picker import (
    create_picker_session,
    poll_session_until_ready,
    list_picked_media_items,
    download_picked_item,
    delete_session,
)
from fastapi.middleware.cors import CORSMiddleware
from storage.local_backend import LocalStorageBackend

app = FastAPI(title="AI Photo Management Platform")

# Allow the frontend (which may run on a different origin) to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Auto-migrate: add created_at column if missing (create_all doesn't add new
# columns to existing tables, so we handle it explicitly).
from sqlalchemy import text as _text
with engine.connect() as _conn:
    _conn.execute(_text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'photos' AND column_name = 'created_at'
            ) THEN
                ALTER TABLE photos ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT now();
            END IF;
        END $$;
    """))
    _conn.commit()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    background_tasks.add_task(ingest_photo, tmp_path, file.filename)
    return {"status": "queued", "filename": file.filename}


@app.get("/photos")
def list_photos(offset: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    total = db.query(Photo).count()
    photos = (
        db.query(Photo)
        .order_by(Photo.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "photos": [{"id": p.id, "source_path": p.source_path} for p in photos],
    }


@app.get("/photos/{photo_id}")
def get_photo(photo_id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    return {
        "id": photo.id,
        "source_path": photo.source_path,
        "storage_ref": photo.storage_ref,
        "storage_backend": photo.storage_backend,
    }


@app.get("/photos/{photo_id}/image")
def get_photo_image(photo_id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    if photo.storage_backend != "local":
        raise HTTPException(status_code=400, detail="Only local storage supported for direct serving")
    # Normalise any stale Windows-style backslash paths so they resolve on Linux
    file_path = photo.storage_ref.replace("\\", "/")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(file_path)


@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    if photo.storage_backend == "local":
        LocalStorageBackend().delete(photo.storage_ref)

    db.delete(photo)
    db.commit()
    return {"status": "deleted"}


@app.get("/search")
def search(q: str, top_k: int = 5):
    results = search_photos_fn(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/jobs/status")
def get_job_status(db: Session = Depends(get_db)):
    """Poll this while /dedup, /faces/cluster, or /faces/reconcile is running
    to see live progress (written by the Celery worker as it runs)."""
    jobs = db.query(JobStatus).all()
    return {
        job.job_name: {
            "status": job.status,
            "processed": job.processed,
            "total": job.total,
            "extra_info": job.extra_info,
            "updated_at": job.updated_at,
        }
        for job in jobs
    }


@app.get("/jobs/{task_id}")
def get_task_status(task_id: str):
    """Look up a specific Celery task by the id returned from /dedup,
    /faces/cluster, or /faces/reconcile. Complements /jobs/status (which is
    keyed by job name and reflects the latest run of each job type)."""
    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
    }


@app.post("/dedup")
def run_dedup():
    """Finds near-duplicate photos using pgvector's index.

    Runs on a Celery worker, not the request thread, so this returns
    immediately with a task id. Resumable: only processes photos where
    dedup_processed is False, so if the worker is interrupted and restarted,
    it picks up where it left off instead of re-scanning everything. Commits
    every 2,000 photos and writes live progress to job_status -- poll
    GET /jobs/status or GET /jobs/{task_id} to watch it run.
    """
    task = run_dedup_task.delay()
    return {"status": "queued", "task_id": task.id}


@app.get("/duplicates")
def list_duplicates(db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    total = db.query(Duplicate).count()
    duplicates = db.query(Duplicate).offset(offset).limit(limit).all()
    results = []
    for d in duplicates:
        photo = db.query(Photo).filter(Photo.id == d.photo_id).first()
        dup_of = db.query(Photo).filter(Photo.id == d.duplicate_of_id).first()
        results.append({
            "photo_id": d.photo_id,
            "photo_image_url": f"/photos/{d.photo_id}/image",
            "duplicate_of_id": d.duplicate_of_id,
            "duplicate_of_image_url": f"/photos/{d.duplicate_of_id}/image",
            "similarity_score": d.similarity_score,
        })
    return {"count": total, "duplicates": results}


@app.post("/faces/cluster")
def run_face_clustering(model: str | None = None):
    """Detects and clusters faces in all unprocessed photos.

    Runs on a Celery worker; returns immediately with a task id. Uses fast
    incremental matching (each new face vs. its single nearest known face),
    which is why POST /faces/reconcile exists as a periodic correction pass --
    see that endpoint's docstring for why incremental matching alone isn't
    enough at scale.

    model: "hog" (default, set via FACE_DETECTION_MODEL env var) or "cnn" for
    better accuracy on angled/profile faces at significantly higher compute
    cost. Pass as a query param to override the default for this run.
    """
    task = run_face_clustering_task.delay(model)
    return {"status": "queued", "task_id": task.id, "model": model or os.getenv("FACE_DETECTION_MODEL", "hog")}


@app.post("/faces/reconcile")
def run_face_reconciliation():
    """Manually triggers the periodic face-cluster reconciliation pass (it
    also runs automatically every RECONCILE_INTERVAL_SECONDS via Celery beat --
    see celery_app.py).

    Incremental clustering in /faces/cluster only ever compares a new face to
    its single nearest known face; it never re-checks a cluster once faces
    have joined it. If two visually similar-but-different people get merged
    early, everything loosely close to either of them can keep joining that
    same cluster with no signal anything went wrong -- on the full LFW stress
    test this produced one 1,056-face cluster containing clearly different
    people. This job re-clusters every cluster above a size threshold from
    scratch with batch DBSCAN at a stricter tolerance and splits it if the
    incremental result doesn't hold up.
    """
    task = reconcile_face_clusters_task.delay()
    return {"status": "queued", "task_id": task.id}


@app.get("/faces/clusters")
def list_face_clusters(db: Session = Depends(get_db)):
    clusters = db.query(FaceCluster).all()
    results = []
    for cluster in clusters:
        faces = db.query(Face).filter(Face.cluster_id == cluster.id).all()
        results.append({
            "cluster_id": cluster.id,
            "label": cluster.label,
            "face_count": len(faces),
            "photo_image_urls": [f"/photos/{f.photo_id}/image" for f in faces],
        })
    results.sort(key=lambda c: c["face_count"], reverse=True)
    return {"cluster_count": len(results), "clusters": results}


@app.post("/photos/{photo_id}/categorize")
def run_categorize(photo_id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo or photo.embedding is None:
        raise HTTPException(status_code=404, detail="Photo not found or has no embedding")

    embedding = np.array(photo.embedding, dtype=np.float32)
    category = categorize_image(embedding, image_path=photo.storage_ref)

    record = Category(photo_id=photo_id, category=category, confidence=1.0)
    db.add(record)
    db.commit()

    return {"photo_id": photo_id, "category": category}


@app.get("/photos/{photo_id}/categories")
def get_categories(photo_id: str, db: Session = Depends(get_db)):
    categories = db.query(Category).filter(Category.photo_id == photo_id).all()
    return [{"category": c.category, "confidence": c.confidence} for c in categories]


@app.post("/google-photos/session")
def create_google_photos_session():
    """Creates a Google Photos picker session.

    Returns a pickerUri the user must open in a browser to select photos.
    Session ID must be passed to /google-photos/session/{id}/import afterward.
    """
    creds = get_drive_credentials()
    session = create_picker_session(creds)
    return {
        "session_id": session["id"],
        "picker_uri": session["pickerUri"],
        "message": "Open picker_uri in a browser, select photos, then call the import endpoint with this session_id"
    }


@app.post("/google-photos/session/{session_id}/import")
def import_google_photos_session(session_id: str, background_tasks: BackgroundTasks):
    """Waits for the user to finish picking, then downloads and ingests every selected photo.

    This blocks until the user finishes picking (or times out after 5 minutes),
    then queues ingestion of each photo as a background task.
    """
    creds = get_drive_credentials()

    session = poll_session_until_ready(creds, session_id)
    media_items = list_picked_media_items(creds, session_id)

    queued = []
    for item in media_items:
        local_path = download_picked_item(creds, item)
        filename = item["mediaFile"].get("filename", f"{item['id']}.jpg")
        background_tasks.add_task(ingest_photo, local_path, filename)
        queued.append(filename)

    delete_session(creds, session_id)

    return {
        "status": "queued",
        "photos_picked": len(media_items),
        "filenames": queued
    }