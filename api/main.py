from fastapi import FastAPI, Depends, HTTPException
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session

from db.database import get_db, engine, Base
from db.models import Photo
from pipeline.ingest import ingest_photo
from pipeline.search import search_photos as search_photos_fn
from storage.local_backend import LocalStorageBackend

app = FastAPI(title="AI Photo Management Platform")

Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


from fastapi import UploadFile, File
import shutil
import tempfile

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


# Add these imports to api/main.py
from db.models import Duplicate, FaceCluster, Face, Category
from pipeline.dedup import find_clip_duplicates
from pipeline.faces import detect_faces, cluster_faces
from pipeline.categorize import categorize_image
import numpy as np


@app.post("/dedup")
def run_dedup(db: Session = Depends(get_db)):
    photos = db.query(Photo).filter(Photo.embedding.isnot(None)).all()

    saved = []
    for photo in photos:
        similar = (
            db.query(Photo)
            .filter(Photo.id != photo.id)
            .filter(Photo.embedding.cosine_distance(photo.embedding) < 0.05)  # distance < 0.05 = similarity > 0.95
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