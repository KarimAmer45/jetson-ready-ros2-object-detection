"""PyTorch detector backends used by the ROS2 node and benchmark script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

import numpy as np


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    """A detector-neutral bounding box."""

    label: str
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: Optional[int] = None

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center_x(self) -> float:
        return self.xmin + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.ymin + self.height / 2.0


class Detector(Protocol):
    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        """Return detections for an OpenCV-style BGR image."""


def resolve_device(device: str):
    """Resolve `auto`, `cpu`, or `cuda` to a torch device."""

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def clamp_detections(detections: Iterable[Detection], width: int, height: int) -> List[Detection]:
    clamped = []
    for det in detections:
        xmin = float(np.clip(det.xmin, 0, width - 1))
        ymin = float(np.clip(det.ymin, 0, height - 1))
        xmax = float(np.clip(det.xmax, 0, width - 1))
        ymax = float(np.clip(det.ymax, 0, height - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        clamped.append(
            Detection(
                label=det.label,
                score=float(det.score),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_id=det.class_id,
            )
        )
    return clamped


class TorchvisionDetector:
    """Torchvision Faster R-CNN detector using COCO pretrained weights."""

    def __init__(
        self,
        device: str = "auto",
        score_threshold: float = 0.5,
        max_detections: int = 50,
        weights: str = "DEFAULT",
    ) -> None:
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        self.torch = torch
        self.device = resolve_device(device)
        self.score_threshold = score_threshold
        self.max_detections = max_detections

        weights_arg = None
        self.categories = COCO_INSTANCE_CATEGORY_NAMES
        if weights and weights.lower() != "none":
            weights_arg = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.categories = list(weights_arg.meta.get("categories", self.categories))

        model_kwargs = {"weights": weights_arg}
        if weights_arg is None:
            model_kwargs["weights_backbone"] = None
        self.model = fasterrcnn_resnet50_fpn(**model_kwargs)
        self.model.eval().to(self.device)

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        height, width = image_bgr.shape[:2]
        image_rgb = image_bgr[..., ::-1].copy()
        tensor = self.torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        with self.torch.inference_mode():
            prediction = self.model([tensor])[0]

        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()

        detections = []
        for box, score, label_id in zip(boxes, scores, labels):
            if float(score) < self.score_threshold:
                continue
            label_index = int(label_id)
            label = (
                self.categories[label_index]
                if 0 <= label_index < len(self.categories)
                else str(label_index)
            )
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    xmin=float(box[0]),
                    ymin=float(box[1]),
                    xmax=float(box[2]),
                    ymax=float(box[3]),
                    class_id=label_index,
                )
            )
            if len(detections) >= self.max_detections:
                break

        return clamp_detections(detections, width, height)


class YoloDetector:
    """Ultralytics YOLO detector backend, useful when a Jetson YOLO model is ready."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        device: str = "auto",
        score_threshold: float = 0.5,
        max_detections: int = 50,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.device = None if device == "auto" else device
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        height, width = image_bgr.shape[:2]
        results = self.model.predict(
            source=image_bgr,
            conf=self.score_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            names = result.names or {}
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_id = int(cls.item())
                detections.append(
                    Detection(
                        label=str(names.get(class_id, class_id)),
                        score=float(conf.item()),
                        xmin=float(xyxy[0].item()),
                        ymin=float(xyxy[1].item()),
                        xmax=float(xyxy[2].item()),
                        ymax=float(xyxy[3].item()),
                        class_id=class_id,
                    )
                )
        return clamp_detections(detections, width, height)


def create_detector(
    backend: str,
    model: str = "",
    device: str = "auto",
    score_threshold: float = 0.5,
    max_detections: int = 50,
) -> Detector:
    normalized = backend.strip().lower()
    if normalized == "torchvision":
        return TorchvisionDetector(
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
            weights=model or "DEFAULT",
        )
    if normalized in {"yolo", "ultralytics"}:
        return YoloDetector(
            model=model or "yolov8n.pt",
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
    raise ValueError(f"Unsupported detector backend: {backend!r}")
