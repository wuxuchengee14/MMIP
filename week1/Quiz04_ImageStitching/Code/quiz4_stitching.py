from pathlib import Path
import cv2
import numpy as np

# 建立輸出與資料目錄
output_dir = Path("Output")
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path("Data")
data_dir.mkdir(exist_ok=True, parents=True)

left_path = data_dir / "left.jpg"
right_path = data_dir / "right.jpg"

if not left_path.exists() or not right_path.exists():
    raise FileNotFoundError("請確認 Data/left.jpg 與 Data/right.jpg 是否存在！")

img1 = cv2.imread(str(left_path))   # 左圖 (基準圖)
img2 = cv2.imread(str(right_path))  # 右圖

# 1. 轉灰階並提取 SIFT 特徵
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 2. KNN 匹配與 Lowe's Ratio Test
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good = []
ratio = 0.75
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append(m)

# 輸出匹配連線圖供驗證
vis_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good[:60], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(str(output_dir / "good_matches.jpg"), vis_matches)

if len(good) < 4:
    raise RuntimeError(f"通過 Ratio Test 的特徵點不足 ({len(good)} < 4)，無法估計 Homography！")

# 3. 關鍵修正：計算右圖 (img2) 變換到左圖 (img1) 的單應性矩陣
# src = img2 (右圖), dst = img1 (左圖)
src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
if H is None:
    raise RuntimeError("RANSAC 計算 Homography 失敗！")

inliers = int(mask.sum()) if mask is not None else 0

# 4. 動態計算畫布尺寸（防止右圖或上下超出邊界）
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 計算右圖四個角經過 H 變換後的坐標
corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

# 左圖的四個角為 [(0,0), (w1,0), (w1,h1), (0,h1)]
all_corners = np.vstack((warped_corners_img2, np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)))

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 若變換後有向左或向上的位移 (負坐標)，建立平移補償矩陣
translation_dist = [-x_min if x_min < 0 else 0, -y_min if y_min < 0 else 0]
H_translation = np.array([
    [1, 0, translation_dist[0]],
    [0, 1, translation_dist[1]],
    [0, 0, 1]
])

canvas_w = x_max - x_min if x_min < 0 else x_max
canvas_h = y_max - y_min if y_min < 0 else y_max

# 5. 透視變換與無接縫平滑融合
# 將右圖投影至延伸後的畫布
warped_img2 = cv2.warpPerspective(img2, H_translation.dot(H), (canvas_w, canvas_h))

# 將左圖移入畫布對應位置
warped_img1 = np.zeros_like(warped_img2)
t_x, t_y = translation_dist
warped_img1[t_y:t_y + h1, t_x:t_x + w1] = img1

# 建立遮罩進行重疊區域融合
mask1 = (warped_img1 > 0).astype(np.float32)
mask2 = (warped_img2 > 0).astype(np.float32)

# 重疊區域取平均，非重疊區域直接保留
overlap_mask = (mask1 * mask2 > 0)
stitched = warped_img1.astype(np.float32) + warped_img2.astype(np.float32)
stitched[overlap_mask] /= 2.0
stitched = np.clip(stitched, 0, 255).astype(np.uint8)

# 儲存結果
cv2.imwrite(str(output_dir / "stitched_result.jpg"), stitched)

print("=" * 45)
print(f"Image 1 特徵點數 : {len(kp1)}")
print(f"Image 2 特徵點數 : {len(kp2)}")
print(f"通過 Ratio Test 數: {len(good)}")
print(f"RANSAC Inliers 數 : {inliers}")
print(f"Inlier Ratio      : {inliers / max(len(good), 1):.2%}")
print(f"輸出影像解析度   : {canvas_w} x {canvas_h}")
print("拼接完成！請至 Output/stitched_result.jpg 查看結果。")
print("=" * 45)