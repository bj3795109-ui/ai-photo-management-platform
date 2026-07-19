from pipeline.embedding import compute_clip_embedding
from pipeline.categorize import categorize_image, CATEGORY_PROMPTS

FIXTURES = "tests/fixtures"


def test_categorize_returns_valid_category():
    emb = compute_clip_embedding(f"{FIXTURES}/red.jpg")
    result = categorize_image(emb)
    assert result in CATEGORY_PROMPTS.keys()


def test_categorize_with_custom_categories():
    emb = compute_clip_embedding(f"{FIXTURES}/circle.jpg")
    custom_categories = {
        "circle": "a photo of a circle shape",
        "square": "a photo of a square shape",
    }
    result = categorize_image(emb, categories=custom_categories)
    assert result in custom_categories.keys()