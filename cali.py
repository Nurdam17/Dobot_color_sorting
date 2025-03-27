import numpy as np
import cv2 as cv
import yaml
import time

# Параметры шахматной доски
CHESSBOARD_CORNER_NUM_X = 8
CHESSBOARD_CORNER_NUM_Y = 6
SQUARE_SIZE_MM = 25
CAMERA_PARAMETERS_OUTPUT_FILE = "cam2.yaml"
REQUIRED_FRAMES = 15

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка 3D координат доски
objp = np.zeros((CHESSBOARD_CORNER_NUM_X * CHESSBOARD_CORNER_NUM_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNER_NUM_X, 0:CHESSBOARD_CORNER_NUM_Y].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints = []

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Не удалось открыть камеру.")
    exit()

last_saved_time = 0
print(f"\n📷 Наведи камеру на шахматную доску. Нужно {REQUIRED_FRAMES} успешных кадров.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка чтения кадра.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), None)

    if found:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), corners2, found)

        # Автосохранение каждые 1.5 секунды
        if time.time() - last_saved_time > 1.5:
            objpoints.append(objp)
            imgpoints.append(corners2)
            last_saved_time = time.time()
            print(f"✅ Углы сохранены ({len(objpoints)}/{REQUIRED_FRAMES})")

    # Прогресс на экране
    cv.putText(frame, f"Снимков: {len(objpoints)}/{REQUIRED_FRAMES}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv.imshow("Калибровка камеры", frame)

    if cv.waitKey(1) & 0xFF == ord('q') or len(objpoints) >= REQUIRED_FRAMES:
        break

cap.release()
cv.destroyAllWindows()

# Калибровка
print("\n📐 Калибровка камеры...")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n📏 Матрица камеры:\n", mtx)
print("\n🔄 Коэффициенты дисторсии:\n", dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\n✅ Средняя ошибка калибровки: {mean_error / len(objpoints)}")

# Сохранение
data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "square_size_mm": SQUARE_SIZE_MM
}
with open(CAMERA_PARAMETERS_OUTPUT_FILE, "w") as f:
    yaml.dump(data, f)

print(f"\n📂 Параметры сохранены в {CAMERA_PARAMETERS_OUTPUT_FILE}")