import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

from yolo_interfaces.msg import ToolDetection, ToolDetectionArray


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # ----------------------------
        # Declare parameters
        # ----------------------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("window_name", "YOLO")
        self.declare_parameter("conf_thresh", 0.25)
        self.declare_parameter("skip_frames", 1)
        self.declare_parameter("max_fps", 0.0)
        self.declare_parameter("image_topic", "/image_raw")

        # ----------------------------
        # Read parameters
        # ----------------------------
        model_path = self.get_parameter("model_path").value
        self.window_name = self.get_parameter("window_name").value
        self.conf_thresh = self.get_parameter("conf_thresh").value
        self.skip_frames = self.get_parameter("skip_frames").value
        self.max_fps = self.get_parameter("max_fps").value
        self.image_topic = self.get_parameter("image_topic").value

        if model_path == "":
            self.get_logger().error("model_path parameter is required!")
            raise RuntimeError("model_path parameter not set")

        # ----------------------------
        # Logging
        # ----------------------------
        self.get_logger().info(f"Loaded model: {model_path}")
        self.get_logger().info(f"Window name: {self.window_name}")
        self.get_logger().info(f"Confidence threshold: {self.conf_thresh}")
        self.get_logger().info(f"Skip frames: {self.skip_frames}")
        self.get_logger().info(f"Max FPS: {self.max_fps}")
        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")

        # ----------------------------
        # CV Bridge
        # ----------------------------
        self.bridge = CvBridge()

        # ----------------------------
        # Frame control
        # ----------------------------
        self.frame_count = 0
        self.last_inference_time = self.get_clock().now()

        # ----------------------------
        # Subscriber (dynamic camera topic)
        # ----------------------------
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # ----------------------------
        # Publisher (dynamic topic)
        # ----------------------------
        topic_name = f"/yolo/{self.window_name.lower()}/detections"
        self.publisher = self.create_publisher(
            ToolDetectionArray,
            topic_name,
            10
        )

        # ----------------------------
        # Load YOLO model
        # ----------------------------
        self.model = YOLO(model_path)

        self.get_logger().info("YOLO Detector Node Started")

    def image_callback(self, msg):

        # ----------------------------
        # Frame skipping
        # ----------------------------
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return

        # ----------------------------
        # FPS limiting
        # ----------------------------
        if self.max_fps > 0:
            now = self.get_clock().now()
            dt = (now - self.last_inference_time).nanoseconds / 1e9
            if dt < (1.0 / self.max_fps):
                return
            self.last_inference_time = now

        # ----------------------------
        # Convert ROS image → OpenCV
        # ----------------------------
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ----------------------------
        # YOLO inference
        # ----------------------------
        results = self.model(frame)[0]

        detection_array = ToolDetectionArray()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Confidence threshold
            if conf < self.conf_thresh:
                continue

            class_name = self.model.names[cls]

            # Draw bounding box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Create detection message
            det = ToolDetection()
            det.class_name = class_name
            det.confidence = conf
            det.x1 = x1
            det.y1 = y1
            det.x2 = x2
            det.y2 = y2

            detection_array.detections.append(det)

        # ----------------------------
        # Publish detections
        # ----------------------------
        self.publisher.publish(detection_array)

        # ----------------------------
        # Display window
        # ----------------------------
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

