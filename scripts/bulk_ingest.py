# scripts/bulk_ingest.py
import os
import sys
import time
from db.database import SessionLocal
from db.models import Photo
from pipeline.embedding import compute_phash, _load_clip
from storage.local_backend import LocalStorageBackend
import torch
from PIL import Image
import numpy as np

BATCH_SIZE = 32


def bulk_ingest(folder_path: str):
    model, preprocess = _load_clip()
    storage = LocalStorageBackend()
    db = SessionLocal()

    image_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_paths)} images. Starting ingestion...")
    start = time.time()
    processed = 0

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_tensors = []
        valid_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors)
        with torch.no_grad():
            embeddings = model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        for path, embedding in zip(valid_paths, embeddings):
            dest_name = os.path.basename(path)
            storage_ref = storage.upload(path, dest_name)
            phash = compute_phash(path)

            photo = Photo(
                source_path=path.replace("\\", "/"),
                storage_backend="local",
                storage_ref=storage_ref,
                hash=phash,
                embedding=embedding.tolist()
            )
            db.add(photo)
            processed += 1

        db.commit()
        elapsed = time.time() - start
        print(f"Processed {processed}/{len(image_paths)} ({elapsed:.1f}s elapsed)")

    db.close()
    print(f"Done. {processed} photos ingested in {time.time() - start:.1f}s")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    bulk_ingest(folder)