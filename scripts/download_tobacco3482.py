from datasets import load_dataset
import os

OUTPUT_DIR = "tobacco3482_images"
CACHE_DIR = "./tobacco3482_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading Tobacco-3482 dataset...")
print("This may take a while depending on your connection.")

dataset = load_dataset(
    "sasa3396/tobacco3482",
    "default",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# Merge all splits (train/test/validation if they exist)
all_images = []

for split in dataset.keys():
    all_images.extend(dataset[split])

print(f"Got {len(all_images)} images. Saving as JPEGs...")

for i, sample in enumerate(all_images):
    img = sample["image"]      # PIL Image

    img.save(os.path.join(OUTPUT_DIR, f"tobacco_{i:06d}.jpg"))

    if (i + 1) % 500 == 0:
        print(f"Saved {i + 1}/{len(all_images)}")

print(f"Done. {len(all_images)} images saved to {OUTPUT_DIR}/")