import json
import time
import numpy as np
from pydobot import Dobot

port = "/dev/tty.usbserial-130"

def load_detections(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Ошибка: Файл detections.json не найден!")
        return None
    except json.JSONDecodeError:
        print("Ошибка: Невозможно прочитать JSON (файл пустой или поврежден).")
        return None

camera_coordinates = load_detections("camera_coordinates1.json")
if not camera_coordinates:
    exit()

device = Dobot(port=port)
pose = device._get_pose()

x, y, z, model_name = camera_coordinates["X_robot_mm"], camera_coordinates["Y_robot_mm"], camera_coordinates["Z_robot_mm"], camera_coordinates["model_name"]
P_camera = np.array([x, y, z])
R_m = np.array(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ]
)

P_robot_to_camera = np.array([262.2, 25, 306])

P_robot = P_robot_to_camera + np.dot(R_m, P_camera)
diff_x = -112
diff_y = -6
diff_z = 0
device.speed(350, 370)
print(P_robot[0])
print(P_robot[1])
print(P_robot[2])
print(device.pose())
device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, 20, 0)
device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, -40, 0)
device.suck(True)
time.sleep(3)
device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, 0, 0)
if model_name == "red":
    device.move_to(186, 133, 0, 0)
    device.move_to(186, 133, 120, 0)
    device.suck(False)
    device.move_to(186, 133, 0, 0)
elif model_name == "yellow":
    device.move_to(240, 50, 0, 0)
    device.move_to(240, 50, 120, 0)
    device.suck(False)
    device.move_to(240, 50, 0, 0)
device.move_to(120, 0, 0, 0)