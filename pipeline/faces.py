import face_recognition
import numpy as np

def detect_faces(image_path: str) -> list[np.ndarray]:
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
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
    