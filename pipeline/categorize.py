from pipeline.embedding import compute_text_embedding
import numpy as np

# Maps each required category to a CLIP-friendly prompt.
# CLIP performs better with natural sentence-like prompts than bare labels,
# so we keep the prompt separate from the canonical label we return to callers.
CATEGORY_PROMPTS = {
    # Required by assignment brief
    "documents": "a photo of a document, ID card, or official paperwork",
    "prescriptions": "a photo of a medical prescription or pharmacy label",
    "receipts": "a photo of a purchase receipt or invoice",
    "people": "a photo of a person or group of people",
    "travel": "a photo taken while traveling, such as a landmark, airport, or scenic destination",
    "pets": "a photo of a pet or animal",

    # Additional categories for richer real-world coverage
    "food": "a photo of food or a meal",
    "screenshots": "a screenshot of a phone, computer, or app interface",
    "vehicles": "a photo of a car, motorcycle, or other vehicle",
    "nature": "a photo of nature, landscape, plants, or scenery",

    # Catch-all, kept last so it's the fallback when nothing else matches well
    "other": "a photo that does not fit any specific category, such as random objects",
}

_category_embedding_cache: dict[str, np.ndarray] = {}


def _get_category_embeddings(categories: dict[str, str]) -> dict[str, np.ndarray]:
    """Computes (and caches) text embeddings for each category prompt.

    Categories are fixed and small in number, so caching avoids recomputing
    the same CLIP text embedding on every single categorize_image() call.
    """
    for label, prompt in categories.items():
        if label not in _category_embedding_cache:
            _category_embedding_cache[label] = compute_text_embedding(prompt)
    return {label: _category_embedding_cache[label] for label in categories}


def categorize_image(image_embedding: np.ndarray, categories: dict[str, str] = None) -> str:
    """Categorizes an image embedding into one of the required categories.

    Returns the canonical category label (e.g. "documents"), not the CLIP prompt.
    """
    if categories is None:
        categories = CATEGORY_PROMPTS

    category_embeddings = _get_category_embeddings(categories)

    labels = list(category_embeddings.keys())
    scores = [np.dot(image_embedding, category_embeddings[label]) for label in labels]

    best_idx = int(np.argmax(scores))
    return labels[best_idx]