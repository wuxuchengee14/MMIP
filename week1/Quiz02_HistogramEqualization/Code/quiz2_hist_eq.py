from pathlib import Path
from time import perf_counter
import cv2
import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("Output")
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path("Data")
data_dir.mkdir(exist_ok=True, parents=True)

input_path = data_dir / "input.jpg"
if not input_path.exists():
    dummy = np.random.normal(loc=80, scale=25, size=(400, 600)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(input_path), dummy)

img = cv2.imread(str(input_path))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(output_dir / "original_gray.png"), gray)

# OpenCV 均衡化
eq_cv = cv2.equalizeHist(gray)
cv2.imwrite(str(output_dir / "equalized_opencv.png"), eq_cv)

# NumPy 均衡化 (CDF 法)
def hist_eq_numpy(img_gray):
    hist = np.bincount(img_gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = np.nonzero(cdf)[0]
    if len(nonzero) == 0:
        return img_gray.copy()
    cdf_min = cdf[nonzero[0]]
    total = img_gray.size
    lut = (cdf - cdf_min) / (total - cdf_min) * 255.0
    lut = np.clip(np.rint(lut), 0, 255).astype(np.uint8)
    return lut[img_gray]

eq_np = hist_eq_numpy(gray)
cv2.imwrite(str(output_dir / "equalized_numpy.png"), eq_np)

# 繪製直方圖
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
axes[0].set_title("Before Equalization")
axes[1].hist(eq_cv.ravel(), bins=256, range=(0, 256), color='black')
axes[1].set_title("After Equalization")
plt.tight_layout()
plt.savefig(str(output_dir / "hist_comparison.png"), dpi=200)
plt.close()

print(f"MAE: {cv2.absdiff(eq_cv, eq_np).mean():.4f}")