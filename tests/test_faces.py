import numpy as np
from pipeline.faces import cluster_faces, match_face_to_cluster

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