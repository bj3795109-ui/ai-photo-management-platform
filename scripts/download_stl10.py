from torchvision.datasets import STL10
from PIL import Image
import numpy as np
import os

OUTPUT_DIR = "stl10_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading STL-10 unlabeled split (100,000 images, ~2.7GB)....")
print("This may take a while depending on connection speed")

dataset = STL10(root="./stl10_data", split="unlabeled", download=True)

print(f"Got {len(dataset)} image. Saving as jpegs....")

for i in range(len(dataset)):
    img, _ = dataset[i]
    
    img.save(os.path.join(OUTPUT_DIR, f"stl10_{i:06d}.jpg"))
    
    if (i+1) % 5000 ==0:
        print(f"Saved {i+1}/{len(dataset)}")
        
print(f"Done. {len(dataset)} images saved to {OUTPUT_DIR}/")