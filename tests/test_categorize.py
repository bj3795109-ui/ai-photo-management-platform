from pipeline.embedding import compute_clip_embedding
from pipeline.categorize import categorize_image

FIXTURES = "tests/fixtures"

def test_categorize_returns_valid_category():
    emb = compute_clip_embedding(f"{FIXTURES}/red.jpg")
    result = categorize_image(emb)
    assert isinstance(result, str)
    assert len(result) > 0
    
def test_categorize_with_custom_categories():
    emb = compute_clip_embedding(f"{FIXTURES}/circle.jpg")
    custom_categories = ["a photo of a circle", "a photo of a square"]
    result = categorize_image(emb, categories=custom_categories)
    assert result in custom_categories