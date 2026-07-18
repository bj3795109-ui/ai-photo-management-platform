from db.database import SessionLocal
from db.models import Photo
from pipeline.embedding import compute_text_embedding

def search_photos(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = compute_text_embedding(query).tolist()
    
    db = SessionLocal()
    try:
        results = (
            db.query(Photo)
            .order_by(Photo.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )
        return [
            {
                "id":p.id, "source_path":p.source_path, "storage_ref": p.storage_ref
            }
            for p in results
        ]
        
    finally:
        db.close()