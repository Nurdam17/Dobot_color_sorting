from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import yaml
import json
import os
import time
from ultralytics import YOLO


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(String, 'dobot_command', 10)

        self.camera_matrix, self.dist_coeffs = self.load_calibration()
        self.fx = self.camera_matrix[0, 0]

        self.image_points = np.array([[337, 210], [587, 210], [337, 460], [587, 460]], dtype=np.float32)
        self.real_world_points = np.array([[0, 0], [150, 0], [0, 150], [150, 150]], dtype=np.float32)
        self.H, _ = cv2.findHomography(self.image_points, self.real_world_points)

        self.zero_x, self.zero_y = self.load_zero_point()
        self.zero_x_mm, self.zero_y_mm = self.pixel_to_world(self.zero_x, self.zero_y)

        self.model = YOLO("/home/aliyanur/damir_ros/src/dobot_sorting_color/yolo_detection_model/runs/detect/train/weights/best.pt")
        self.cap = cv2.VideoCapture(2)

        if not self.cap.isOpened():
            self.get_logger().error("Camera not found!")
            exit()

        self.timer = self.create_timer(0.1, self.detect_and_publish)

        self.detected_file = "/home/aliyanur/damir_ros/src/dobot_sorting_color/dobot_sorting_color/detected_objects.json"
        self.max_objects = 7
        self.last_detection_time = time.time()
        self.last_detection_count = 0
        self.publish_count = 0
        self.handled_keys = set()
        self.first_cycle = True

        with open(self.detected_file, "w") as f:
            json.dump([], f)

    def load_zero_point(self):
        with open("/home/aliyanur/damir_ros/src/dobot_sorting_color/dobot_sorting_color/zero_points1.json", "r") as f:
            data = json.load(f)
        return data["zero_x"], data["zero_y"]

    def load_calibration(self):
        calib_file = "/home/aliyanur/damir_ros/src/dobot_sorting_color/dobot_sorting_color/cam2.yaml"
        with open(calib_file, "r") as f:
            data = yaml.safe_load(f)
        return np.array(data["camera_matrix"]), np.array(data["dist_coeff"]).ravel()

    def pixel_to_world(self, x_pixel, y_pixel):
        pt = np.array([[x_pixel, y_pixel, 1]], dtype=np.float32).T
        world = np.dot(self.H, pt)
        world /= world[2]
        return world[0][0], world[1][0]

    def detect_and_publish(self):
        if not os.path.exists(self.detected_file):
            with open(self.detected_file, "w") as f:
                json.dump([], f)

        with open(self.detected_file, "r") as f:
            data = json.load(f)

        if len(data) > 0:
            obj_data = data.pop(0)
            with open(self.detected_file, "w") as f:
                json.dump(data, f, indent=2)

            msg = String()
            msg.data = json.dumps(obj_data)
            self.publisher.publish(msg)
            self.get_logger().info(f"Published: {msg.data}")

            self.publish_count += 1
            if self.publish_count >= self.max_objects:
                self.get_logger().info("All objects published. Resetting cycle.")
                self.publish_count = 0
                self.last_detection_count = 0
                with open(self.detected_file, "w") as f:
                    json.dump([], f)
                time.sleep(1.5)
            return

        delay = 1 if self.first_cycle else 6 * max(self.last_detection_count + 1, 2)
        elapsed = time.time() - self.last_detection_time
        remaining = max(0, delay - elapsed)
        if elapsed < delay:
            self.get_logger().info(f"Waiting {remaining:.1f} sec before next detection... ({elapsed:.1f}/{delay} sec elapsed)")
            return

        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning("Frame not read.")
            return

        results = self.model(frame)
        cube_width_mm = 25
        detected_objects = []
        new_keys = set()

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for i, box in enumerate(result.boxes.xyxy):
                x_min, y_min, x_max, y_max = map(int, box[:4])
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min

                if width < 20:
                    continue

                real_x, real_y = self.pixel_to_world(x_center, y_center)
                real_x -= self.zero_x_mm
                real_y -= self.zero_y_mm
                z_mm = (cube_width_mm * self.fx) / width

                key = f"{int(round(-real_x / 5.0) * 5)}_{int(round(real_y / 5.0) * 5)}_{int(round(z_mm / 5.0) * 5)}"
                if key in self.handled_keys or key in new_keys:
                    continue

                class_index = int(result.boxes.cls[i])
                class_name = self.model.names[class_index]

                obj_data = {
                    "model_name": class_name,
                    "X_robot_mm": -real_x,
                    "Y_robot_mm": real_y,
                    "Z_robot_mm": z_mm
                }
                detected_objects.append(obj_data)
                new_keys.add(key)
                self.get_logger().info(f"Detected: {obj_data}")

                if len(detected_objects) >= self.max_objects:
                    break

        if len(detected_objects) > 0:
            self.get_logger().info(f"Saving {len(detected_objects)} unique object(s) to JSON.")
            self.last_detection_time = time.time()
            self.last_detection_count = len(detected_objects)
            self.handled_keys.update(new_keys)
            self.first_cycle = False
            with open(self.detected_file, "w") as f:
                json.dump(detected_objects, f, indent=2)


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
