import numpy as np
import cv2 as cv
import yaml
from pathlib import Path

# Количество внутренних углов шахматной доски (не количество клеток!)
CHESSBOARD_CORNER_NUM_X = 8
CHESSBOARD_CORNER_NUM_Y = 6
SQUARE_SIZE_MM = 25  # 🔹 Размер одной клетки в мм (например, 25 мм)

IMAGE_SRC = "cam_images"
CAMERA_PARAMETERS_OUTPUT_FILE = "cam1.yaml"

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set path to the images
calib_imgs_path = root.joinpath(IMAGE_SRC)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготавливаем реальные 3D координаты углов шахматной доски
objp = np.zeros((CHESSBOARD_CORNER_NUM_X * CHESSBOARD_CORNER_NUM_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNER_NUM_X, 0:CHESSBOARD_CORNER_NUM_Y].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM  # 🔹 Теперь координаты учитывают реальный размер клеток

# Массивы для хранения точек
objpoints = []  # 3D точки (в реальном пространстве)
imgpoints = []  # 2D точки (в пикселях)

images = calib_imgs_path.glob('*.JPG')  # Берем все изображения в формате JPG

for fname in images:
    img = cv.imread(str(root.joinpath(fname)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Поиск углов шахматной доски
    ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), None)

    # Если нашли углы - сохраняем их
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Рисуем найденные углы
        cv.drawChessboardCorners(img, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(50)
    else:
        print(f'⚠️ Шахматная доска не найдена в {fname}')

cv.destroyAllWindows()

print("📷 Калибруем камеру...")

# Калибровка камеры
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n📏 Матрица камеры:\n", mtx)
print("\n🔄 Коэффициенты дисторсии:\n", dist)

# Вычисляем среднюю ошибку калибровки
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\n✅ Средняя ошибка калибровки: {mean_error / len(objpoints)}")

# Сохраняем калибровочные параметры в YAML
data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "square_size_mm": SQUARE_SIZE_MM
}

with open(CAMERA_PARAMETERS_OUTPUT_FILE, "w") as f:
    yaml.dump(data, f)

print(f"\n📂 Параметры сохранены в {CAMERA_PARAMETERS_OUTPUT_FILE}")