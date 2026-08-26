"""One-time migration: normalise Windows backslash paths in the photos table.

Usage (inside the Docker container):
    python -m scripts.fix_paths

Safe to run multiple times — it's a no-op if paths are already clean.
"""
from db.database import SessionLocal
from db.models import Photo


def fix_paths():
    db = SessionLocal()
    try:
        photos = db.query(Photo).all()
        storage_fixed = 0
        source_fixed = 0

        for photo in photos:
            changed = False

            if "\\" in photo.storage_ref:
                photo.storage_ref = photo.storage_ref.replace("\\", "/")
                storage_fixed += 1
                changed = True

            if "\\" in photo.source_path:
                photo.source_path = photo.source_path.replace("\\", "/")
                source_fixed += 1
                changed = True

        db.commit()
        print(f"[fix_paths] storage_ref: {storage_fixed} rows fixed")
        print(f"[fix_paths] source_path: {source_fixed} rows fixed")
        print(f"[fix_paths] Done. ({len(photos)} total photos scanned)")
    finally:
        db.close()


if __name__ == "__main__":
    fix_paths()
