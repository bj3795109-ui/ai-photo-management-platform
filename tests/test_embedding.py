import numpy as np
from pipeline.embedding import(
    compute_phash,
    hamming_distance,
    compute_clip_embedding,
    compute_text_embedding
)

FIXTURES = "tests/fixtures"

def test_phash_different_images_have_zero_distance():
    h1 = compute_phash(f"{FIXTURES}/red.jpg")
    h2 = compute_phash(f"{FIXTURES}/red_copy.jpg")
    assert hamming_distance(h1, h2) == 0
    
def test_phash_different_images_have_nonzero_distance():
    h1 = compute_phash(f"{FIXTURES}/red.jpg")
    h2 = compute_phash(f"{FIXTURES}/blue.jpg")
    assert hamming_distance(h1, h2) > 0
    
def test_clip_embedding_shape_and_normalization():
    emb = compute_clip_embedding(f"{FIXTURES}/red.jpg")
    assert emb.shape ==(512,)
    assert emb.dtype == np.float32
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
    
def test_text_embedding_shape_and_normalization():
    emb = compute_text_embedding("a red square")
    assert emb.shape ==(512, )
    assert np.isclose(np.linalg.norm(emb), 1.0,atol=1e-3)
    