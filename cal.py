import numpy as np
import cv2 as cv
import yaml

# 🔹 Параметры шахматной доски
CHESSBOARD_CORNER_NUM_X = 8  # Внутренние углы по X
CHESSBOARD_CORNER_NUM_Y = 6  # Внутренние углы по Y
SQUARE_SIZE_MM = 25  # Размер клетки в мм
CAMERA_PARAMETERS_OUTPUT_FILE = "cam1.yaml"

# 🔹 Настройки поиска углов
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 🔹 Подготовка 3D координат шахматной доски (реальный мир)
objp = np.zeros((CHESSBOARD_CORNER_NUM_X * CHESSBOARD_CORNER_NUM_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNER_NUM_X, 0:CHESSBOARD_CORNER_NUM_Y].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM  # Учитываем размер клеток

# 🔹 Массивы для хранения точек
objpoints = []  # 3D точки в реальном пространстве
imgpoints = []  # 2D точки в пикселях

# 🔹 Открываем камеру
cap = cv.VideoCapture(0)  # Индекс 0 - основная камера

if not cap.isOpened():
    print("❌ Ошибка: не удалось открыть камеру!")
    exit()

print("\n📷 Найди шахматную доску в кадре и нажми 's' для сохранения точки.")
print("🚀 Для завершения калибровки и сохранения параметров нажми 'q'.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка: не удалось считать кадр.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 🔹 Поиск шахматной доски
    found, corners = cv.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), None)

    if found:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), corners2, found)

    cv.imshow("Camera Calibration", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('s') and found:
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"✅ Углы сохранены! Текущие кадры: {len(objpoints)}")

    if key == ord('q') and len(objpoints) > 0:
        break

cap.release()
cv.destroyAllWindows()

# 🔹 Калибровка камеры
print("📷 Калибруем камеру...")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n📏 Матрица камеры:\n", mtx)
print("\n🔄 Коэффициенты дисторсии:\n", dist)

# 🔹 Вычисляем среднюю ошибку калибровки
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\n✅ Средняя ошибка калибровки: {mean_error / len(objpoints)}")

# 🔹 Сохраняем калибровочные параметры в YAML
data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "square_size_mm": SQUARE_SIZE_MM
}

with open(CAMERA_PARAMETERS_OUTPUT_FILE, "w") as f:
    yaml.dump(data, f)

print(f"\n📂 Параметры сохранены в {CAMERA_PARAMETERS_OUTPUT_FILE}")