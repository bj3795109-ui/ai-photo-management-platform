import imagehash
from PIL import Image
import numpy as np

def compute_phash(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img))

def hamming_distance(hash1: str, hash2: str) -> int:
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    
    return h1-h2

import torch
import open_clip

_model = None
_preprocess = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_clip():
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        _model.to(_device)
        _model.eval()
    return _model, _preprocess

def compute_clip_embedding(image_path: str) -> np.ndarray:
    model, preprocess = _load_clip()
    
    img = Image.open(image_path).convert("RGB")
    
    img_tensor = preprocess(img).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.cpu().numpy().squeeze().astype(np.float32)
    
    return embedding

def compute_text_embedding(text: str) -> np.ndarray:
    model, preprocess = _load_clip()
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    tokens = tokenizer([text]).to(_device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.cpu().numpy().squeeze().astype(np.float32)
    
    return embedding

