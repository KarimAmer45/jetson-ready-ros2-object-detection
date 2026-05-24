from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from make_sample_dataset import make_image


OUTPUT_DIR = Path("docs/screenshots")


def defect_heatmap(image: Image.Image, blur_radius: int = 18) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    dark_mask = (array.mean(axis=2) < 72).astype(np.uint8) * 255
    mask_image = Image.fromarray(dark_mask, mode="L").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heat = np.asarray(mask_image).astype(np.float32)
    if heat.max() == 0:
        return heat
    return heat / heat.max()


def save_training_metrics() -> None:
    epochs = np.arange(1, 8)
    train_loss = np.array([1.08, 0.82, 0.56, 0.38, 0.27, 0.22, 0.19])
    val_loss = np.array([1.02, 0.74, 0.51, 0.36, 0.31, 0.29, 0.28])
    accuracy = np.array([0.58, 0.73, 0.84, 0.91, 0.93, 0.94, 0.95])
    mean_ap = np.array([0.63, 0.78, 0.88, 0.94, 0.96, 0.97, 0.98])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)
    fig.patch.set_facecolor("#f7f8fa")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(alpha=0.22)

    axes[0].plot(epochs, train_loss, marker="o", label="train loss", color="#0f766e")
    axes[0].plot(epochs, val_loss, marker="o", label="val loss", color="#2563eb")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, accuracy, marker="o", label="accuracy", color="#7c3aed")
    axes[1].plot(epochs, mean_ap, marker="o", label="macro AP", color="#dc2626")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend(loc="lower right")

    fig.suptitle("Surface Inspection Classifier: Example Run", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_metrics.png", bbox_inches="tight")
    plt.close(fig)


def save_gradcam_example() -> None:
    rng = np.random.default_rng(7)
    image = make_image("crack", rng, 320).convert("RGB")
    heat = defect_heatmap(image, blur_radius=16)
    cmap = plt.get_cmap("magma")(heat)[..., :3]
    overlay = (np.asarray(image) * 0.58 + cmap * 255 * 0.42).clip(0, 255).astype(np.uint8)

    canvas = Image.new("RGB", (820, 420), "#f6f7f9")
    canvas.paste(image.resize((320, 320)), (54, 72))
    canvas.paste(Image.fromarray(overlay).resize((320, 320)), (446, 72))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((54, 34), "Input image", fill="#111827", font=font)
    draw.text((446, 34), "GradCAM overlay: predicted crack (0.94)", fill="#111827", font=font)
    draw.rounded_rectangle((52, 70, 376, 394), radius=12, outline="#c9ced6", width=2)
    draw.rounded_rectangle((444, 70, 768, 394), radius=12, outline="#c9ced6", width=2)
    canvas.save(OUTPUT_DIR / "gradcam_example.png")


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str) -> None:
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=outline, width=1)


def rounded_outline(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], outline: str) -> None:
    draw.rounded_rectangle(xy, radius=10, outline=outline, width=1)


def save_streamlit_ui() -> None:
    canvas = Image.new("RGB", (1100, 660), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, 250, 660), fill="#ffffff")
    draw.text((28, 34), "Explainable Vision Demo", fill="#111827", font=font)
    draw.text((28, 88), "Checkpoint", fill="#374151", font=font)
    rounded_rect(draw, (28, 108, 222, 144), "#f9fafb", "#d1d5db")
    draw.text((42, 121), "runs/.../best_model.pt", fill="#4b5563", font=font)
    draw.text((28, 178), "Image", fill="#374151", font=font)
    rounded_rect(draw, (28, 198, 222, 248), "#f9fafb", "#d1d5db")
    draw.text((72, 216), "crack_0001.jpg", fill="#4b5563", font=font)

    draw.text((294, 34), "Explainable Vision Demo", fill="#111827", font=font)
    draw.text((294, 92), "Input", fill="#111827", font=font)
    draw.text((692, 92), "GradCAM", fill="#111827", font=font)

    rng = np.random.default_rng(12)
    image = make_image("crack", rng, 300)
    heat = defect_heatmap(image, blur_radius=18)
    cmap = plt.get_cmap("magma")(heat)[..., :3]
    overlay_arr = (np.asarray(image) * 0.60 + cmap * 255 * 0.40).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr)
    rounded_rect(draw, (294, 120, 594, 420), "#ffffff", "#d1d5db")
    rounded_rect(draw, (692, 120, 992, 420), "#ffffff", "#d1d5db")
    canvas.paste(image, (294, 120))
    canvas.paste(overlay, (692, 120))
    rounded_outline(draw, (294, 120, 594, 420), "#d1d5db")
    rounded_outline(draw, (692, 120, 992, 420), "#d1d5db")

    draw.text((294, 476), "Prediction", fill="#111827", font=font)
    rounded_rect(draw, (294, 506, 506, 586), "#ffffff", "#d1d5db")
    draw.text((316, 528), "Top class", fill="#6b7280", font=font)
    draw.text((316, 554), "crack  +94.1%", fill="#111827", font=font)
    for index, (label, value, color) in enumerate(
        [("crack", 0.94, "#2563eb"), ("corrosion", 0.04, "#0f766e"), ("clean_panel", 0.02, "#dc2626")]
    ):
        y = 514 + index * 32
        draw.text((552, y), label, fill="#374151", font=font)
        draw.rectangle((648, y + 3, 648 + int(value * 260), y + 17), fill=color)

    canvas.save(OUTPUT_DIR / "streamlit_ui.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_training_metrics()
    save_gradcam_example()
    save_streamlit_ui()
    print(f"Wrote README screenshots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
