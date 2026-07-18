from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np
import os

OUTPUT_DIR = "lfw_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading LFW dataset (this may take a few minuteson first run)....")
lfw = fetch_lfw_people(color=True, min_faces_per_person=1, resize=1.0)

print(f"Got {len(lfw.images)} images. Savingas jpegs...")
for i, img_array in enumerate(lfw.images):
    img = Image.fromarray((img_array*255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_DIR, f"lfw_{i:05d}.jpg"))
    
print(f"Done. {len(lfw.images)} images saved to {OUTPUT_DIR}/")