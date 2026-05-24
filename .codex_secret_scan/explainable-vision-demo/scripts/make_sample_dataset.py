from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASSES = ("clean_panel", "corrosion", "crack")


def base_panel(rng: np.random.Generator, image_size: int) -> Image.Image:
    base = rng.normal(loc=176, scale=13, size=(image_size, image_size, 3)).clip(95, 230)
    gradient = np.linspace(-12, 16, image_size).reshape(1, image_size, 1)
    base = (base + gradient).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(rng.integers(1, 4)):
        y = int(rng.integers(20, image_size - 20))
        draw.line((0, y, image_size, y + int(rng.integers(-2, 3))), fill=(80, 92, 100, 28), width=2)
    return image.filter(ImageFilter.GaussianBlur(radius=0.35))


def add_corrosion(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for _ in range(rng.integers(5, 11)):
        cx = int(rng.integers(25, width - 25))
        cy = int(rng.integers(25, height - 25))
        rx = int(rng.integers(8, 28))
        ry = int(rng.integers(6, 24))
        color = (
            int(rng.integers(120, 190)),
            int(rng.integers(58, 95)),
            int(rng.integers(22, 42)),
            int(rng.integers(75, 145)),
        )
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
        draw.ellipse((cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2), fill=(90, 45, 20, 55))
    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


def add_crack(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    points = []
    x = int(rng.integers(18, width // 3))
    y = int(rng.integers(20, height - 20))
    for _ in range(rng.integers(7, 12)):
        points.append((x, y))
        x += int(rng.integers(14, 30))
        y += int(rng.integers(-24, 25))
        y = max(15, min(height - 15, y))
    draw.line(points, fill=(25, 30, 34, 220), width=int(rng.integers(3, 6)), joint="curve")

    for point in points[2:-2:2]:
        branch = [
            point,
            (
                point[0] + int(rng.integers(10, 32)),
                point[1] + int(rng.choice([-1, 1]) * rng.integers(12, 34)),
            ),
        ]
        draw.line(branch, fill=(28, 32, 36, 150), width=2)
    return image


def make_image(class_name: str, rng: np.random.Generator, image_size: int) -> Image.Image:
    image = base_panel(rng, image_size)
    if class_name == "corrosion":
        image = add_corrosion(image, rng)
    elif class_name == "crack":
        image = add_crack(image, rng)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic surface-inspection dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--images-per-class", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    rng = np.random.default_rng(args.seed)
    for class_name in CLASSES:
        class_dir = output / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.images_per_class):
            image = make_image(class_name, rng, args.image_size)
            image.save(class_dir / f"{class_name}_{index:04d}.jpg", quality=92)
    print(f"Wrote {args.images_per_class * len(CLASSES)} images to {output}")


if __name__ == "__main__":
    main()
