import os
import shutil
import tempfile

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.database import get_db, engine, Base
from db.models import Photo, Duplicate, FaceCluster, Face, Category
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


@app.post("/dedup")
def run_dedup(db: Session = Depends(get_db)):
    photos = db.query(Photo).filter(Photo.embedding.isnot(None)).all()

    saved = []
    for photo in photos:
        similar = (
            db.query(Photo)
            .filter(Photo.id != photo.id)
            .filter(Photo.embedding.cosine_distance(photo.embedding) < 0.05)
            .limit(5)
            .all()
        )
        for match in similar:
            existing = db.query(Duplicate).filter(
                Duplicate.photo_id == photo.id,
                Duplicate.duplicate_of_id == match.id
            ).first()
            if not existing:
                record = Duplicate(photo_id=photo.id, duplicate_of_id=match.id, similarity_score=1.0)
                db.add(record)
                saved.append({"photo_id": photo.id, "duplicate_of": match.id})

    db.commit()
    return {"duplicates_found": len(saved), "duplicates": saved}


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
    unprocessed_photos = db.query(Photo).filter(Photo.faces_processed == False).all()

    if not unprocessed_photos:
        return {"message": "No new photos to process", "faces_found": 0}

    existing_faces = db.query(Face).all()
    known_encodings = [np.array(f.embedding, dtype=np.float32) for f in existing_faces]
    known_cluster_ids = [f.cluster_id for f in existing_faces]

    faces_found = 0
    for photo in unprocessed_photos:
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

    db.commit()
    return {"faces_found": faces_found, "photos_processed": len(unprocessed_photos)}


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