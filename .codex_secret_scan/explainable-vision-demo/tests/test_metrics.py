import numpy as np

from xai_vision_demo.metrics import classification_metrics


def test_classification_metrics_multiclass_are_bounded():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.65, 0.15],
            [0.10, 0.20, 0.70],
            [0.55, 0.30, 0.15],
            [0.25, 0.60, 0.15],
            [0.15, 0.20, 0.65],
        ]
    )

    metrics = classification_metrics(y_true, y_prob, ["clean_panel", "corrosion", "crack"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_auc_ovr"] is not None
    assert 0.0 <= metrics["macro_auc_ovr"] <= 1.0
    assert metrics["macro_average_precision"] is not None
    assert 0.0 <= metrics["macro_average_precision"] <= 1.0
