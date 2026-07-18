from pipeline.embedding import compute_phash, compute_clip_embedding
from pipeline.dedup import find_phash_duplicates, find_clip_duplicates

FIXTURES = "tests/fixtures"

def test_find_phash_duplicates_detects_near_duplicate():
    hashes = {
        "red": compute_phash(f"{FIXTURES}/red.jpg"),
        "red_copy":compute_phash(f"{FIXTURES}/red_copy.jpg"),
        "blue": compute_phash(f"{FIXTURES}/blue.jpg"),
    }
    duplicates = find_phash_duplicates(hashes)
    pairs = [(d[0], d[1]) for d in duplicates]
    assert ("red", "red_copy") in pairs
    
def test_find_phash_duplicates_no_false_positive():
    hashes = {
        "red": compute_phash(f"{FIXTURES}/red.jpg"),
        "blue": compute_phash(f"{FIXTURES}/blue.jpg"),
    }
    duplicates = find_phash_duplicates(hashes)
    assert duplicates ==[]
    
def test_find_clip_duplicates_detects_near_duplicate():
    embeddings = {
        "red": compute_clip_embedding(f"{FIXTURES}/red.jpg"),
        "red_copy": compute_clip_embedding(f"{FIXTURES}/red_copy.jpg"),
        "blue": compute_clip_embedding(f"{FIXTURES}/blue.jpg"),
    }
    duplicates = find_clip_duplicates(embeddings, threshold=0.9)
    pairs = [(d[0], d[1]) for d in duplicates]
    assert ("red", "red_copy") in pairs
    