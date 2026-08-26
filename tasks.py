from celery_app import celery_app
from pipeline.jobs import run_dedup_job, run_face_clustering_job, run_reconciliation_job


@celery_app.task(name="tasks.run_dedup")
def run_dedup_task():
    return run_dedup_job()


@celery_app.task(name="tasks.run_face_clustering")
def run_face_clustering_task(model: str | None = None):
    return run_face_clustering_job(model=model)


@celery_app.task(name="tasks.reconcile_face_clusters")
def reconcile_face_clusters_task():
    """Runs on the schedule defined in celery_app.py's beat_schedule, and can
    also be triggered manually via POST /faces/reconcile."""
    return run_reconciliation_job()
