"""Long-running job logic, extracted from api/main.py so it can be executed
by Celery workers instead of inline on the request thread.

Each *_job() function opens and closes its own DB session, since it no longer
runs inside a FastAPI request (no Depends(get_db) available). This is the
same logic that used to live directly in the /dedup and /faces/cluster route
handlers -- moving it here is what let those routes become thin wrappers that
just call task.delay() and return immediately.
"""
import os
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func

from db.database import SessionLocal
from db.models import Photo, Duplicate, FaceCluster, Face, JobStatus
from pipeline.faces import detect_faces, cluster_faces, plan_cluster_reassignment

RECONCILE_SIZE_THRESHOLD = int(os.getenv("RECONCILE_SIZE_THRESHOLD", "30"))
RECONCILE_TOLERANCE = float(os.getenv("RECONCILE_TOLERANCE", "0.5"))


def update_job_status(db, job_name: str, processed: int, total: int, status: str, extra_info: str = None):
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


def run_dedup_job() -> dict:
    """Finds near-duplicate photos using pgvector's index.

    Resumable: only processes photos where dedup_processed is False, so if
    interrupted and restarted, it picks up where it left off instead of
    re-scanning everything from the start. Commits every 2,000 photos (not
    just once at the end), and writes progress to job_status (poll
    GET /jobs/status to see it live).
    """
    db = SessionLocal()
    try:
        photos = (
            db.query(Photo.id, Photo.embedding)
            .filter(Photo.embedding.isnot(None))
            .filter(Photo.dedup_processed == False)
            .all()
        )
        total = len(photos)

        if total == 0:
            update_job_status(db, "dedup", 0, 0, "done", "No new photos to process")
            return {"message": "No new photos to process", "duplicates_found": 0}

        existing_pairs = set(
            (d.photo_id, d.duplicate_of_id)
            for d in db.query(Duplicate.photo_id, Duplicate.duplicate_of_id).all()
        )

        saved_count = 0
        update_job_status(db, "dedup", 0, total, "running", "0 duplicates found so far")

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
                update_job_status(db, "dedup", i + 1, total, "running", f"{saved_count} duplicates found so far")
                print(f"[dedup] Processed {i + 1}/{total} photos, {saved_count} duplicates found so far")

        db.commit()
        update_job_status(db, "dedup", total, total, "done", f"{saved_count} duplicates found total")
        print(f"[dedup] Done. Processed {total} photos, {saved_count} duplicates found total")
        return {"duplicates_found": saved_count, "photos_processed": total}
    finally:
        db.close()


def run_face_clustering_job(model: str | None = None) -> dict:
    """Detects and clusters faces in all unprocessed photos.

    Commits every 500 photos processed (not just once at the end) so an
    interruption doesn't lose all progress. Progress is written to job_status.

    This is the fast, incremental path: each new face is matched against only
    its single nearest known face. It never re-validates a cluster once faces
    have joined it, which is exactly the gap run_reconciliation_job() below
    exists to close on a periodic basis.
    """
    db = SessionLocal()
    try:
        unprocessed_photos = db.query(Photo).filter(Photo.faces_processed == False).all()
        total = len(unprocessed_photos)

        if not unprocessed_photos:
            update_job_status(db, "faces_cluster", 0, 0, "done", "No new photos to process")
            return {"message": "No new photos to process", "faces_found": 0}

        existing_faces = db.query(Face).all()
        known_encodings = [np.array(f.embedding, dtype=np.float32) for f in existing_faces]
        known_cluster_ids = [f.cluster_id for f in existing_faces]

        faces_found = 0
        update_job_status(db, "faces_cluster", 0, total, "running", "0 faces found so far")

        for i, photo in enumerate(unprocessed_photos):
            # Normalise any stale Windows-style backslash paths for Linux
            image_path = photo.source_path.replace("\\", "/")
            encodings = detect_faces(image_path, model=model)
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
                update_job_status(db, "faces_cluster", i + 1, total, "running", f"{faces_found} faces found so far")
                print(f"[faces/cluster] Processed {i + 1}/{total} photos, {faces_found} faces found so far")

        db.commit()
        update_job_status(db, "faces_cluster", total, total, "done", f"{faces_found} faces found total")
        print(f"[faces/cluster] Done. Processed {total} photos, {faces_found} faces found total")
        return {"faces_found": faces_found, "photos_processed": total}
    finally:
        db.close()


def run_reconciliation_job(size_threshold: int = RECONCILE_SIZE_THRESHOLD, tolerance: float = RECONCILE_TOLERANCE) -> dict:
    """Periodic correction pass for incremental face clustering.

    The incremental matcher in run_face_clustering_job() only ever compares a
    new face to its single nearest known face -- it never re-checks whether a
    cluster still makes sense once faces have joined it. If two visually
    similar-but-different people get merged early, everything loosely close to
    either of them can keep joining that same cluster with no signal that
    anything went wrong (this is exactly what produced a 1,056-face cluster
    of clearly different people on the full LFW stress test).

    This job re-examines every cluster with at least `size_threshold` faces,
    re-clusters that cluster's own faces from scratch with batch DBSCAN at a
    stricter tolerance, and splits the cluster if DBSCAN disagrees with the
    incremental result. The largest resulting group keeps the original
    cluster id (and any existing label); every other group is split into a
    new cluster. Small/coherent clusters are left untouched -- this only
    targets clusters big enough for drift to plausibly have occurred.
    """
    db = SessionLocal()
    clusters_examined = 0
    clusters_split = 0
    faces_reassigned = 0

    try:
        large_cluster_ids = [
            row[0]
            for row in (
                db.query(FaceCluster.id)
                .join(Face, Face.cluster_id == FaceCluster.id)
                .group_by(FaceCluster.id)
                .having(func.count(Face.id) >= size_threshold)
                .all()
            )
        ]

        update_job_status(db, "face_reconciliation", 0, len(large_cluster_ids), "running",
                           f"0/{len(large_cluster_ids)} large clusters checked")

        for idx, cluster_id in enumerate(large_cluster_ids):
            faces = db.query(Face).filter(Face.cluster_id == cluster_id).all()
            clusters_examined += 1

            embeddings = [np.array(f.embedding, dtype=np.float32) for f in faces]
            labels = cluster_faces(embeddings, tolerance=tolerance)
            keep_label, needs_split = plan_cluster_reassignment(labels)

            if needs_split:
                label_to_cluster_id = {keep_label: cluster_id}
                for label in set(labels):
                    if label == keep_label:
                        continue
                    new_cluster = FaceCluster(label=None)
                    db.add(new_cluster)
                    db.flush()
                    label_to_cluster_id[label] = new_cluster.id

                for face, label in zip(faces, labels):
                    target_id = label_to_cluster_id[label]
                    if face.cluster_id != target_id:
                        face.cluster_id = target_id
                        faces_reassigned += 1

                clusters_split += 1
                db.commit()
                print(f"[reconciliation] Split cluster {cluster_id}: {len(set(labels))} groups found "
                      f"({len(faces)} faces re-examined)")

            update_job_status(
                db, "face_reconciliation", idx + 1, len(large_cluster_ids), "running",
                f"{idx + 1}/{len(large_cluster_ids)} large clusters checked, {clusters_split} split so far",
            )

        update_job_status(
            db, "face_reconciliation", len(large_cluster_ids), len(large_cluster_ids), "done",
            f"{clusters_examined} clusters examined, {clusters_split} split, {faces_reassigned} faces reassigned",
        )
        print(f"[reconciliation] Done. {clusters_examined} clusters examined, "
              f"{clusters_split} split, {faces_reassigned} faces reassigned")

        return {
            "clusters_examined": clusters_examined,
            "clusters_split": clusters_split,
            "faces_reassigned": faces_reassigned,
        }
    finally:
        db.close()
