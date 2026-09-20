from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

output_dir = Path("Output")
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path("Data")
data_dir.mkdir(exist_ok=True, parents=True)

input_path = data_dir / "input.jpg"
if not input_path.exists():
    dummy = np.zeros((400, 600, 3), dtype=np.uint8)
    dummy[:, :, 0] = np.linspace(0, 255, 600, dtype=np.uint8)
    dummy[:, :, 1] = np.linspace(0, 255, 400, dtype=np.uint8)[:, None]
    dummy[:, :, 2] = 128
    cv2.imwrite(str(input_path), dummy)

img = cv2.imread(str(input_path))

# OpenCV 灰階
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(output_dir / "gray_opencv.png"), gray_cv)

# NumPy 灰階 (ITU-R BT.601, BGR 順序)
b = img[:, :, 0].astype(np.float32)
g = img[:, :, 1].astype(np.float32)
r = img[:, :, 2].astype(np.float32)
gray_np = np.clip(np.rint(0.114 * b + 0.587 * g + 0.299 * r), 0, 255).astype(np.uint8)
cv2.imwrite(str(output_dir / "gray_numpy.png"), gray_np)

# 誤差分析
abs_diff = cv2.absdiff(gray_cv, gray_np)
vis_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(str(output_dir / "difference.png"), vis_diff)

# 速度測試
N = 500
t0 = perf_counter()
for _ in range(N):
    _ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_time = (perf_counter() - t0) / N * 1000

t0 = perf_counter()
for _ in range(N):
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    _ = np.clip(np.rint(0.114 * b + 0.587 * g + 0.299 * r), 0, 255).astype(np.uint8)
np_time = (perf_counter() - t0) / N * 1000

print(f"OpenCV: {cv_time:.4f} ms | NumPy: {np_time:.4f} ms | MAE: {np.mean(abs_diff):.4f} | MaxDiff: {np.max(abs_diff)}")