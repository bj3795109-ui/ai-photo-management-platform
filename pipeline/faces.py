import os
from collections import Counter

import face_recognition
import numpy as np

DEFAULT_FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")


def detect_faces(image_path: str, model: str | None = None) -> list[np.ndarray]:
    """Detects faces and returns their embeddings.

    model: "hog" (default) is fast but less reliable on angled/profile faces.
    "cnn" is slower (and much slower without a GPU) but meaningfully more
    accurate on non-frontal faces. Falls back to the FACE_DETECTION_MODEL env
    var, then "hog", if not passed explicitly.
    """
    resolved_model = model or DEFAULT_FACE_DETECTION_MODEL
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=resolved_model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_encodings

from sklearn.cluster import DBSCAN

def cluster_faces(face_encodings: list, tolerance: float = 0.6) -> list[int]:
    if not face_encodings:
        return []
    
    encodings_array = np.array(face_encodings)
    
    clustering = DBSCAN(
        metric = "euclidean",
        eps = tolerance,
        min_samples = 1
    ).fit(encodings_array)
    
    return clustering.labels_.tolist()

def match_face_to_cluster(
    new_encoding: np.ndarray,
    known_encodings: list[np.ndarray],
    known_labels: list[int],
    tolerance: float = 0.6,
) -> int:
    
    if not known_encodings:
        return 0
    
    known_array = np.array(known_encodings)
    distances = np.linalg.norm(known_array- new_encoding, axis=1)
    
    min_distance = distances.min()
    best_match_idx = distances.argmin()
    
    if min_distance <= tolerance:
        return known_labels[best_match_idx]
    
    return max(known_labels) + 1


def plan_cluster_reassignment(labels: list[int]) -> tuple[int, bool]:
    """Decides how to correct one existing cluster during periodic reconciliation.

    `labels` is the output of re-running cluster_faces() (batch DBSCAN) over
    every face currently in one FaceCluster -- i.e. asking "does this cluster
    still look coherent under a strict, from-scratch re-check?" This is the
    piece that's missing from incremental clustering: match_face_to_cluster()
    only ever compares one new face to its nearest neighbor and never
    revisits that decision, so a bad early merge is never caught. This
    function is what a periodic reconciliation pass uses to catch it.

    Returns (keep_label, needs_split):
    - needs_split is False if DBSCAN still finds one group -- the cluster is
      left untouched.
    - Otherwise keep_label is the largest resulting group, which keeps the
      cluster's original id (and label, if any); every other group should be
      moved to a newly created cluster.
    """
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return (next(iter(unique_labels)) if unique_labels else 0), False

    keep_label = Counter(labels).most_common(1)[0][0]
    return keep_label, True
