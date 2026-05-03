import numpy as np

from object_detection_ros2.detectors import Detection, clamp_detections


def test_detection_geometry_properties():
    detection = Detection(
        label="person",
        score=0.9,
        xmin=10,
        ymin=20,
        xmax=70,
        ymax=100,
        class_id=1,
    )

    assert detection.width == 60
    assert detection.height == 80
    assert detection.center_x == 40
    assert detection.center_y == 60


def test_clamp_detections_filters_invalid_boxes():
    detections = [
        Detection("car", 0.8, -10, 5, 200, 90),
        Detection("bad", 0.5, 40, 40, 35, 80),
    ]

    clamped = clamp_detections(detections, width=100, height=80)

    assert len(clamped) == 1
    assert clamped[0].xmin == 0
    assert clamped[0].xmax == 99
    assert clamped[0].ymax == 79
