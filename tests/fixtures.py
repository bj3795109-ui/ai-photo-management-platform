from PIL import Image, ImageDraw
import random
import os

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


def make_textured(name, base_color, seed):
    random.seed(seed)
    img = Image.new("RGB", (200, 200), color=base_color)
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = random.randint(0, 200), random.randint(0, 200)
        r = random.randint(5, 20)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))
    img.save(os.path.join(FIXTURES_DIR, name))


if __name__ == "__main__":
    make_textured("red.jpg", (255, 0, 0), seed=1)
    make_textured("red_copy.jpg", (255, 0, 0), seed=1)
    make_textured("blue.jpg", (0, 0, 255), seed=2)
    make_textured("circle.jpg", (0, 200, 0), seed=3)
    print("Fixtures created in", FIXTURES_DIR)