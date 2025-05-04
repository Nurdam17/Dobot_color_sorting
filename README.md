# Dobot Color Sorting Robot Arm (ROS 2)

## Project Description
This project implements an automated color-based sorting system using a **Dobot robotic arm** integrated with **ROS 2**. The system detects colored objects using a camera and **YOLOv8** for object detection, calculates their real-world coordinates via homography and calibration, and commands the Dobot to pick and place the objects accordingly.

**Main components:**
- **camera.py:** Publishes images from the camera.
- **main.py:** Subscribes to image data, performs detection, and sends commands to Dobot.
- **calibration.py:** Tools for camera calibration and homography setup.

## Requirements
- ROS 2 Humble or newer
- Python 3.8+
- OpenCV
- ultralytics (YOLOv8)
- Dobot SDK / Python API
- numpy, PyYAML

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nurdam17/Dobot_color_sorting.git
   cd Dobot_color_sorting
   ```

2. Build the ROS 2 package:
   ```bash
   colcon build --packages-select dobot_sorting_color
   source install/setup.bash
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(or manually install: `opencv-python`, `ultralytics`, `numpy`, etc.)*

## Usage
1. **Calibrate the camera:**
   - Run `calibration.py` to calibrate intrinsic and extrinsic parameters using a chessboard.

2. **Start camera publisher:**
   ```bash
   ros2 run dobot_sorting_color camera.py
   ```

3. **Run the main controller:**
   ```bash
   ros2 run dobot_sorting_color main.py
   ```

4. **Launch script:**
   Alternatively, use the provided script:
   ```bash
   ./launch_pick_place.sh
   ```

## Project Structure
```
dobot_sorting_color/
├── camera.py
├── main.py
├── calibration.py
├── cam1.yaml (camera config)
├── zero_point.json (reference point)
├── yolo_detection_model/
│   ├── yolov8n.pt (YOLOv8 model)
│   └── runs/ (training logs)
├── detected_objects.json (output)
└── ...
```

## Notes
- The YOLO model used is `yolov8n.pt`. You can replace it with a custom-trained model as needed.
- Make sure the Dobot arm is properly connected and the SDK is installed.
- Coordinate transformation is handled via homography; ensure good calibration for precision.

## Contributing
Pull requests and improvements are welcome. Please submit issues if you find bugs or have suggestions!

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
**Author:** Nurdam17

