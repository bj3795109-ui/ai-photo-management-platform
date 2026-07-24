import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.database import get_db, engine, Base
from db.models import Photo, Duplicate, FaceCluster, Face, Category, JobStatus
from pipeline.ingest import ingest_photo
from pipeline.search import search_photos as search_photos_fn
from pipeline.faces import detect_faces
from pipeline.categorize import categorize_image
from auth import get_drive_credentials
from sources.google_photos_picker import (
    create_picker_session,
    poll_session_until_ready,
    list_picked_media_items,
    download_picked_item,
    delete_session,
)
from storage.local_backend import LocalStorageBackend

app = FastAPI(title="AI Photo Management Platform")

Base.metadata.create_all(bind=engine)


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
def list_photos(db: Session = Depends(get_db)):
    photos = db.query(Photo).limit(50).all()
    return [{"id": p.id, "source_path": p.source_path} for p in photos]


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
    if not os.path.exists(photo.storage_ref):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(photo.storage_ref)


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


def _update_job_status(db: Session, job_name: str, processed: int, total: int, status: str, extra_info: str = None):
    """Upserts a row in job_status. Commits immediately so it's visible to other
    requests polling GET /jobs/status while this job is still running."""
    job = db.query(JobStatus).filter(JobStatus.job_name == job_name).first()
    if job is None:
        job = JobStatus(job_name=job_name)
        db.add(job)

    job.processed = processed
    job.total = total
    job.status = status
    job.extra_info = extra_info
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


@app.get("/jobs/status")
def get_job_status(db: Session = Depends(get_db)):
    """Poll this while /dedup or /faces/cluster is running to see live progress."""
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


@app.post("/dedup")
def run_dedup(db: Session = Depends(get_db)):
    """Finds near-duplicate photos using pgvector's index.

    Resumable: only processes photos where dedup_processed is False, so if
    interrupted and restarted, it picks up where it left off instead of
    re-scanning everything from the start. Commits every 2,000 photos (not
    just once at the end), and writes progress to job_status (poll
    GET /jobs/status from a separate request to see it live).
    """
    photos = (
        db.query(Photo.id, Photo.embedding)
        .filter(Photo.embedding.isnot(None))
        .filter(Photo.dedup_processed == False)
        .all()
    )
    total = len(photos)

    if total == 0:
        _update_job_status(db, "dedup", 0, 0, "done", "No new photos to process")
        return {"message": "No new photos to process", "duplicates_found": 0}

    existing_pairs = set(
        (d.photo_id, d.duplicate_of_id)
        for d in db.query(Duplicate.photo_id, Duplicate.duplicate_of_id).all()
    )

    saved_count = 0
    _update_job_status(db, "dedup", 0, total, "running", "0 duplicates found so far")

    for i, (photo_id, embedding) in enumerate(photos):
        similar = (
            db.query(Photo.id)
            .filter(Photo.id != photo_id)
            .filter(Photo.embedding.cosine_distance(embedding) < 0.05)
            .limit(5)
            .all()
        )
        for (match_id,) in similar:
            pair = (photo_id, match_id)
            if pair not in existing_pairs:
                db.add(Duplicate(photo_id=photo_id, duplicate_of_id=match_id, similarity_score=1.0))
                existing_pairs.add(pair)
                saved_count += 1

        db.query(Photo).filter(Photo.id == photo_id).update({"dedup_processed": True})

        if (i + 1) % 2000 == 0:
            db.commit()
            _update_job_status(db, "dedup", i + 1, total, "running", f"{saved_count} duplicates found so far")
            print(f"[dedup] Processed {i + 1}/{total} photos, {saved_count} duplicates found so far")

    db.commit()
    _update_job_status(db, "dedup", total, total, "done", f"{saved_count} duplicates found total")
    print(f"[dedup] Done. Processed {total} photos, {saved_count} duplicates found total")
    return {"duplicates_found": saved_count, "photos_processed": total}


@app.get("/duplicates")
def list_duplicates(db: Session = Depends(get_db), limit: int = 50):
    duplicates = db.query(Duplicate).limit(limit).all()
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
    return {"count": len(results), "duplicates": results}


@app.post("/faces/cluster")
def run_face_clustering(db: Session = Depends(get_db)):
    """Detects and clusters faces in all unprocessed photos.

    Commits every 500 photos processed (not just once at the end) so an
    interruption doesn't lose all progress. Progress is written to job_status
    (poll GET /jobs/status from a separate request to see it live) and printed
    to server logs.
    """
    unprocessed_photos = db.query(Photo).filter(Photo.faces_processed == False).all()
    total = len(unprocessed_photos)

    if not unprocessed_photos:
        _update_job_status(db, "faces_cluster", 0, 0, "done", "No new photos to process")
        return {"message": "No new photos to process", "faces_found": 0}

    existing_faces = db.query(Face).all()
    known_encodings = [np.array(f.embedding, dtype=np.float32) for f in existing_faces]
    known_cluster_ids = [f.cluster_id for f in existing_faces]

    faces_found = 0
    _update_job_status(db, "faces_cluster", 0, total, "running", "0 faces found so far")

    for i, photo in enumerate(unprocessed_photos):
        encodings = detect_faces(photo.source_path)
        for enc in encodings:
            matched_cluster_id = None

            if known_encodings:
                distances = [np.linalg.norm(np.array(k) - enc) for k in known_encodings]
                min_dist = min(distances)
                if min_dist <= 0.6:
                    matched_cluster_id = known_cluster_ids[distances.index(min_dist)]

            if matched_cluster_id is None:
                cluster = FaceCluster(label=None)
                db.add(cluster)
                db.flush()
                matched_cluster_id = cluster.id

            face = Face(photo_id=photo.id, bbox={}, embedding=enc.tolist(), cluster_id=matched_cluster_id)
            db.add(face)
            known_encodings.append(enc)
            known_cluster_ids.append(matched_cluster_id)
            faces_found += 1

        photo.faces_processed = True

        if (i + 1) % 500 == 0:
            db.commit()
            _update_job_status(db, "faces_cluster", i + 1, total, "running", f"{faces_found} faces found so far")
            print(f"[faces/cluster] Processed {i + 1}/{total} photos, {faces_found} faces found so far")

    db.commit()
    _update_job_status(db, "faces_cluster", total, total, "done", f"{faces_found} faces found total")
    print(f"[faces/cluster] Done. Processed {total} photos, {faces_found} faces found total")
    return {"faces_found": faces_found, "photos_processed": total}


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
    category = categorize_image(embedding)

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