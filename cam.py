import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import json
import os

ZERO_POINT_FILE = "zero_points1.json"

def load_zero_point():
    if os.path.exists(ZERO_POINT_FILE):
        with open(ZERO_POINT_FILE, "r") as f:
            data = json.load(f)
            return data["zero_x"], data["zero_y"]
    return None, None

def load_calibration_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["camera_matrix"]), np.array(data["dist_coeff"]).ravel()

camera_matrix, dist_coeffs = load_calibration_yaml("cam2.yaml")
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

image_points = np.array([
    [963, 551],  # Top-left
    [1513, 551],  # Top-right
    [963, 1001],  # Bottom-left
    [1513, 1001]   # Bottom-right
], dtype=np.float32)

real_world_points = np.array([
    [0, 0],      # Top-left in mm
    [150, 0],    # Top-right
    [0, 125],    # Bottom-left
    [150, 125]   # Bottom-right
], dtype=np.float32)


H, _ = cv2.findHomography(image_points, real_world_points)

def pixel_to_world_homography(x_pixel, y_pixel, H):
    pixel_point = np.array([[x_pixel, y_pixel, 1]], dtype=np.float32).T
    world_point = np.dot(H, pixel_point)
    world_point /= world_point[2]  
    return world_point[0][0], world_point[1][0] 

zero_x, zero_y = load_zero_point()

if zero_x is None or zero_y is None:
    print("Error")
    exit()

model = YOLO("runs/detect/train/weights/best.pt")
cube_width_mm = 24.7
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error")
        break
    camera_coordinates = {}

    results = model(frame)

    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue

        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            perceived_width = x_max - x_min

            if perceived_width < 10:
                print("Error")
                continue

            real_x_mm, real_y_mm = pixel_to_world_homography(x_center, y_center, H)

            zero_x_mm, zero_y_mm = pixel_to_world_homography(zero_x, zero_y, H)
            real_x_mm -= zero_x_mm
            real_y_mm -= zero_y_mm

            distance_mm = (cube_width_mm * fx) / perceived_width

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distance_mm:.0f} mm", (x_min, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"X: {real_x_mm:.0f} mm", (x_min, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Y: {real_y_mm:.0f} mm", (x_min, y_min - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            class_index = int(result.boxes.cls[0]) 
            class_name = model.names[class_index]  
            camera_coordinates = {
                "model_name": class_name,
                "X_robot_mm": real_x_mm,
                "Y_robot_mm": real_y_mm,
                "Z_robot_mm": distance_mm
            }

    if camera_coordinates:
        with open("camera_coordinates1.json", "w") as f:
            json.dump(camera_coordinates, f, indent=4)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()