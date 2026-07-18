from pipeline.embedding import compute_text_embedding
import numpy as np

DEFAULT_CATEGORIES = [
    "a photo of a person", "a photo of food", "a photo of a document or screenshot",
    "a photo of nature or landscape", "a photo of a pet or animal",
    "a photo of a building or architecture", "a photo taken at a party or event",
    "a photo of a vehicle", "a selfie"
]

def categorize_image(image_embedding: np.ndarray, categories: list[str]=None) ->str:
    if categories is None:
        categories = DEFAULT_CATEGORIES
        
    scores = []
    for category in categories:
        text_emb = compute_text_embedding(category)
        similarity = np.dot(image_embedding, text_emb)
        scores.append(similarity)
        
    best_idx = int(np.argmax(scores))
    return categories[best_idx]