from pathlib import Path
import cv2
import numpy as np

output_dir = Path("Output")
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path("Data")
data_dir.mkdir(exist_ok=True, parents=True)

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))

sample_path = data_dir / "sample01.jpg"
if not sample_path.exists():
    canvas = np.full((600, 800, 3), 50, dtype=np.uint8)
    doc_pts = np.array([[180, 120], [620, 160], [540, 500], [140, 440]], dtype=np.int32)
    cv2.fillPoly(canvas, [doc_pts], (230, 230, 230))
    cv2.putText(canvas, "MMIP PERSPECTIVE TEST", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.imwrite(str(sample_path), canvas)

img = cv2.imread(str(sample_path))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
cv2.imwrite(str(output_dir / "edges.jpg"), edges)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
quad = None
for c in contours[:10]:
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
        break

if quad is None:
    quad = np.array([[180, 120], [620, 160], [540, 500], [140, 440]], dtype=np.float32)

warped = four_point_transform(img, quad)
cv2.imwrite(str(output_dir / "warped_result.jpg"), warped)
print("Quiz 3 完成")