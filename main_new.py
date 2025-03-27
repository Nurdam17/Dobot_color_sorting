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
if not camera_coordinates or not isinstance(camera_coordinates, list):
    print("Ошибка: Неверный формат данных.")
    exit()

device = Dobot(port=port)
device.speed(350, 370)

R_m = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1]
])
P_robot_to_camera = np.array([262.2, 25, 306])

# Смещения
diff_x = -97
diff_y = -10
diff_z = 0

for obj in camera_coordinates:
    x, y, z = obj["X_robot_mm"], obj["Y_robot_mm"], obj["Z_robot_mm"]
    model_name = obj["model_name"]

    P_camera = np.array([x, y, z])
    P_robot = P_robot_to_camera + np.dot(R_m, P_camera)

    print(f"\nОбъект: {model_name}")
    print(f"Координаты (робот): X={P_robot[0]:.1f}, Y={P_robot[1]:.1f}, Z={P_robot[2]:.1f}")

    # Движения захвата
    device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, 30, 0)
    device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, -40, 0)
    device.suck(True)
    time.sleep(3)
    device.move_to(P_robot[0] + diff_x, P_robot[1] + diff_y, 30, 0)

    # Размещение по цвету
    if model_name == "red":
        device.move_to(214, 142, 30, 0)
        device.move_to(214, 142, 0, 0)
        device.suck(False)
        device.move_to(214, 142, 30, 0)
    elif model_name == "yellow":
        device.move_to(208, 85, 30, 0)
        device.move_to(208, 85, 0, 0)
        device.suck(False)
        device.move_to(208, 85, 30, 0)

    device.move_to(120, 0, 0, 0)

    # Пауза между объектами
    time.sleep(5)

print("Обработка завершена.")