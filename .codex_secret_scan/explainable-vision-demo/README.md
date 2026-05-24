# Explainable Object Detection / Classification Demo

An end-to-end computer vision demo for image classification with explainability. The repo uses a transfer-learning-ready ResNet18/EfficientNet pipeline, reproducible dataset splits, accuracy/AUC/classification-mAP evaluation, GradCAM visualizations, and a small Streamlit UI for interactive inference.

The included sample dataset generator creates a tiny synthetic surface-inspection dataset so the full workflow can run without external data. The same training and evaluation code also works with any ImageFolder-style dataset.

## Training output

![Training metrics](docs/screenshots/training_metrics.png)

![GradCAM example](docs/screenshots/gradcam_example.png)

![Streamlit UI](docs/screenshots/streamlit_ui.png)

## Explainability workflow

- Reproducible image dataset discovery and stratified `train` / `val` / `test` splitting.
- ResNet18 or EfficientNet-B0 classification heads for transfer learning or scratch training.
- Training loop with validation tracking, checkpointing, and learning-curve exports.
- Evaluation with accuracy, macro ROC-AUC, and macro average precision as classification mAP.
- GradCAM overlays that highlight image regions driving the predicted class.
- Streamlit UI for uploading an image, viewing top predictions, and inspecting GradCAM.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Create a small demo dataset:

```bash
python scripts/make_sample_dataset.py --output data/surface_inspection --images-per-class 80
```

Train a classifier:

```bash
python -m xai_vision_demo.train \
  --data-dir data/surface_inspection \
  --output-dir runs/surface_resnet18 \
  --arch resnet18 \
  --epochs 5 \
  --batch-size 16 \
  --freeze-backbone
```

To fine-tune from ImageNet weights, add `--pretrained`. That may download torchvision weights the first time.

Evaluate the test split:

```bash
python -m xai_vision_demo.evaluate \
  --checkpoint runs/surface_resnet18/best_model.pt \
  --split-csv runs/surface_resnet18/splits.csv \
  --output-dir runs/surface_resnet18/eval
```

Create a GradCAM overlay for one image:

```bash
python -m xai_vision_demo.explain \
  --checkpoint runs/surface_resnet18/best_model.pt \
  --image data/surface_inspection/crack/crack_0001.jpg \
  --output runs/surface_resnet18/gradcam_crack.png
```

Launch the UI:

```bash
streamlit run app/streamlit_app.py
```

## Dataset Format

Use an ImageFolder layout:

```text
data/my_dataset/
  class_a/
    image_001.jpg
  class_b/
    image_002.jpg
  class_c/
    image_003.jpg
```

The training command writes `splits.csv`, `classes.json`, `metrics_history.json`, `training_curves.png`, and `best_model.pt` to the selected run directory.

## Metrics

The evaluator reports:

- `accuracy`: top-1 accuracy.
- `macro_auc_ovr`: macro one-vs-rest ROC-AUC for multi-class classification.
- `macro_average_precision`: macro average precision. This is the classification analogue of mAP and is useful when comparing ranked confidence scores.

## Dataset limitations

- The bundled dataset is intentionally synthetic and small; use a real domain dataset before drawing product conclusions.
- GradCAM is a localization aid, not a proof of causal reasoning.
- Object detection is not implemented in this repo. A YOLO variant could be added with bounding-box labels and detection mAP.
- The Streamlit UI loads a local checkpoint only; model registry or cloud deployment integration would be the next production step.
- Add calibration metrics, test-time augmentation, and failure-case galleries for a stronger model audit.
