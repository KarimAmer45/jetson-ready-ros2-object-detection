"""ROS2 node that subscribes to camera frames and publishes detections."""

from __future__ import annotations

import time
from typing import Iterable

from cv_bridge import CvBridge
import cv2
from diagnostic_msgs.msg import KeyValue
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray

try:
    from vision_msgs.msg import ObjectHypothesisWithPose
except ImportError:  # pragma: no cover - only for uncommon vision_msgs variants
    ObjectHypothesisWithPose = None

from object_detection_ros2.detectors import Detection, create_detector


class ObjectDetectionNode(Node):
    """Run a PyTorch detector for every subscribed camera image."""

    def __init__(self) -> None:
        super().__init__("object_detector")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("annotated_topic", "/detections/annotated")
        self.declare_parameter("stats_topic", "/detections/stats")
        self.declare_parameter("backend", "torchvision")
        self.declare_parameter("model", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("score_threshold", 0.5)
        self.declare_parameter("max_detections", 50)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("queue_size", 2)
        self.declare_parameter("log_interval_sec", 5.0)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        stats_topic = self.get_parameter("stats_topic").value
        backend = self.get_parameter("backend").value
        model = self.get_parameter("model").value
        device = self.get_parameter("device").value
        score_threshold = float(self.get_parameter("score_threshold").value)
        max_detections = int(self.get_parameter("max_detections").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self.bridge = CvBridge()
        self.detector = create_detector(
            backend=backend,
            model=model,
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.last_log_time = time.perf_counter()
        self.frame_count = 0
        self.latency_ms_accumulator = 0.0

        self.detections_pub = self.create_publisher(
            Detection2DArray, detections_topic, queue_size
        )
        self.annotated_pub = self.create_publisher(Image, annotated_topic, queue_size)
        self.stats_pub = self.create_publisher(KeyValue, stats_topic, queue_size)
        self.subscription = self.create_subscription(
            Image, image_topic, self.on_image, queue_size
        )

        self.get_logger().info(
            "Object detector ready: "
            f"backend={backend}, model={model or 'default'}, device={device}, "
            f"image_topic={image_topic}"
        )

    def on_image(self, msg: Image) -> None:
        started = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.detector.predict(frame)
        latency_ms = (time.perf_counter() - started) * 1000.0

        self.detections_pub.publish(self.to_detection_array(msg, detections))
        if self.publish_annotated:
            annotated = draw_detections(frame.copy(), detections)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

        self.publish_stats(latency_ms, len(detections))

    def to_detection_array(
        self, msg: Image, detections: Iterable[Detection]
    ) -> Detection2DArray:
        array = Detection2DArray()
        array.header = msg.header
        array.detections = [to_detection_msg(det) for det in detections]
        return array

    def publish_stats(self, latency_ms: float, detection_count: int) -> None:
        self.frame_count += 1
        self.latency_ms_accumulator += latency_ms

        now = time.perf_counter()
        elapsed = now - self.last_log_time
        if elapsed < self.log_interval_sec:
            return

        fps = self.frame_count / elapsed
        mean_latency = self.latency_ms_accumulator / max(self.frame_count, 1)
        value = (
            f"fps={fps:.2f}, latency_ms_mean={mean_latency:.2f}, "
            f"last_detections={detection_count}"
        )
        self.stats_pub.publish(KeyValue(key="object_detector", value=value))
        self.get_logger().info(value)

        self.last_log_time = now
        self.frame_count = 0
        self.latency_ms_accumulator = 0.0


def to_detection_msg(det: Detection) -> Detection2D:
    msg = Detection2D()
    msg.bbox = BoundingBox2D()
    set_bbox_center(msg.bbox, det.center_x, det.center_y)
    msg.bbox.size_x = float(det.width)
    msg.bbox.size_y = float(det.height)
    if hasattr(msg, "id"):
        msg.id = det.label

    if ObjectHypothesisWithPose is not None:
        hypothesis = ObjectHypothesisWithPose()
        if hasattr(hypothesis, "hypothesis"):
            hypothesis.hypothesis.class_id = str(
                det.class_id if det.class_id is not None else det.label
            )
            hypothesis.hypothesis.score = float(det.score)
        else:
            hypothesis.id = str(det.class_id if det.class_id is not None else det.label)
            hypothesis.score = float(det.score)
        msg.results.append(hypothesis)
    return msg


def set_bbox_center(bbox: BoundingBox2D, x: float, y: float) -> None:
    center = bbox.center
    if hasattr(center, "position"):
        center.position.x = float(x)
        center.position.y = float(y)
        return
    center.x = float(x)
    center.y = float(y)


def draw_detections(image, detections: Iterable[Detection]):
    for det in detections:
        pt1 = (int(det.xmin), int(det.ymin))
        pt2 = (int(det.xmax), int(det.ymax))
        cv2.rectangle(image, pt1, pt2, (50, 220, 120), 2)

        label = f"{det.label} {det.score:.2f}"
        text_origin = (pt1[0], max(18, pt1[1] - 8))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (text_origin[0], text_origin[1] - text_height - baseline - 4),
            (text_origin[0] + text_width + 6, text_origin[1] + baseline),
            (50, 220, 120),
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (text_origin[0] + 3, text_origin[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
