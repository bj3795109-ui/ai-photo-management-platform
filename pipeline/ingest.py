from db.database import SessionLocal
from db.models import Photo
from pipeline.embedding import compute_clip_embedding, compute_phash
from storage.local_backend import LocalStorageBackend

def ingest_photo(local_file_path: str, dest_name: str) -> str:
    storage = LocalStorageBackend()
    storage_ref = storage.upload(local_file_path, dest_name)
    
    embedding = compute_clip_embedding(local_file_path)
    phash = compute_phash(local_file_path)
    
    db = SessionLocal()
    try:
        photo = Photo(
            source_path=local_file_path.replace("\\", "/"),
            storage_backend="local",
            storage_ref=storage_ref,
            hash=phash,
            embedding=embedding.tolist()
        )
        db.add(photo)
        db.commit()
        db.refresh(photo)
        return photo.id
    
    finally:
        db.close()
    