"""
Wrapper that serves the existing FastAPI app + the frontend UI.

Usage:
    python serve_ui.py

This imports the original app from api.main (untouched) and adds:
  1. CORS middleware (so the frontend can call API endpoints)
  2. A static mount for the frontend/ directory
  3. A catch-all fallback that serves index.html for SPA routing

The original api/main.py is NOT modified in any way.
"""

import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import the existing app — this also runs Base.metadata.create_all
from api.main import app

# --- CORS (allow the frontend to talk to the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from db.database import get_db
from db.models import Photo
from sqlalchemy.orm import Session
from fastapi import Depends

# --- Serve frontend files ---
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


from typing import Optional
from db.models import Duplicate

@app.get("/api/photos/count", include_in_schema=False)
def get_photos_count(db: Session = Depends(get_db)):
    return {"count": db.query(Photo).count()}

@app.get("/api/duplicates", include_in_schema=False)
def get_duplicates_paginated(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    count = db.query(Duplicate).count()
    duplicates = db.query(Duplicate).limit(limit).offset(offset).all()
    results = []
    for d in duplicates:
        photo_url = f"/photos/{d.photo_id}/image"
        dup_url = f"/photos/{d.duplicate_of_id}/image"
        results.append({
            "photo_id": d.photo_id,
            "duplicate_of_id": d.duplicate_of_id,
            "similarity_score": d.similarity_score,
            "photo_image_url": photo_url,
            "duplicate_of_image_url": dup_url
        })
    return {"count": count, "duplicates": results}





@app.get("/", include_in_schema=False)
def serve_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount static files AFTER the root route so API routes take priority
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
