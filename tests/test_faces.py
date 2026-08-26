import numpy as np
from pipeline.faces import cluster_faces, match_face_to_cluster, plan_cluster_reassignment, detect_faces

def make_fake_encoding(seed: int)-> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float32)

def test_cluster_faces_groups_similar_encodings():
    base = make_fake_encoding(1)
    similar = base + np.random.default_rng(2).normal(0,0.01,128).astype(np.float32)
    different = make_fake_encoding(99)
    
    labels = cluster_faces([base, similar,different], tolerance=0.3)
    
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
    
    
def test_cluster_faces_empty_input():
    assert cluster_faces([]) == []
    
def test_match_face_to_cluster_reuses_existing_label():
    known = [make_fake_encoding(1)]
    labels = [0]
    similar = known[0] + np.random.default_rng(3).normal(0,0.01,128).astype(np.float32)
    
    result = match_face_to_cluster(similar, known, labels, tolerance=0.3)
    assert result == 0
    
def test_match_face_to_cluster_assigns_new_label_for_different_face():
    known = [make_fake_encoding(1)]
    labels = [0]
    different = make_fake_encoding(99)
    
    result = match_face_to_cluster(different, known, labels, tolerance=0.3)
    assert result==1


def test_plan_cluster_reassignment_no_split_for_coherent_cluster():
    # batch DBSCAN agreeing on one group -> reconciliation should leave it alone
    keep_label, needs_split = plan_cluster_reassignment([0, 0, 0, 0])
    assert needs_split is False
    assert keep_label == 0


def test_plan_cluster_reassignment_splits_when_multiple_groups_found():
    # reproduces the failure mode: an incrementally-built cluster that batch
    # DBSCAN, re-checked from scratch, disagrees with -- 5 faces in one group,
    # 2 faces that drifted in as a distinct group
    labels = [0, 0, 0, 0, 0, 1, 1]
    keep_label, needs_split = plan_cluster_reassignment(labels)
    assert needs_split is True
    assert keep_label == 0  # the larger group keeps the original cluster id


def test_plan_cluster_reassignment_empty_input():
    keep_label, needs_split = plan_cluster_reassignment([])
    assert needs_split is False
    assert keep_label == 0


def test_detect_faces_accepts_model_override(monkeypatch):
    # verifies the model param is threaded through to face_recognition rather
    # than always using the hard-coded default
    calls = {}

    def fake_face_locations(image, model):
        calls["model"] = model
        return []

    monkeypatch.setattr("pipeline.faces.face_recognition.load_image_file", lambda path: np.zeros((10, 10, 3)))
    monkeypatch.setattr("pipeline.faces.face_recognition.face_locations", fake_face_locations)
    monkeypatch.setattr("pipeline.faces.face_recognition.face_encodings", lambda image, locations: [])

    detect_faces("fake_path.jpg", model="cnn")
    assert calls["model"] == "cnn"


def test_detect_faces_defaults_to_hog(monkeypatch):
    calls = {}

    def fake_face_locations(image, model):
        calls["model"] = model
        return []

    monkeypatch.setattr("pipeline.faces.face_recognition.load_image_file", lambda path: np.zeros((10, 10, 3)))
    monkeypatch.setattr("pipeline.faces.face_recognition.face_locations", fake_face_locations)
    monkeypatch.setattr("pipeline.faces.face_recognition.face_encodings", lambda image, locations: [])

    detect_faces("fake_path.jpg")
    assert calls["model"] == "hog"