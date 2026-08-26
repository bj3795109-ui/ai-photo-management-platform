import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# How often the face-cluster reconciliation pass runs, in seconds. Defaults to
# every 6 hours -- frequent enough that a bad merge doesn't compound for long,
# infrequent enough not to constantly re-scan every large cluster. Override
# with RECONCILE_INTERVAL_SECONDS (e.g. set low for local demoing).
RECONCILE_INTERVAL_SECONDS = float(os.getenv("RECONCILE_INTERVAL_SECONDS", str(6 * 60 * 60)))

celery_app = Celery(
    "photo_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # /dedup and /faces/cluster are long CPU-bound jobs (minutes to hours at
    # 100k+ images) -- ack after completion, not on receipt, so a crashed
    # worker doesn't silently drop an in-flight job.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.beat_schedule = {
    "face-cluster-reconciliation": {
        "task": "tasks.reconcile_face_clusters",
        "schedule": RECONCILE_INTERVAL_SECONDS,
    },
}
