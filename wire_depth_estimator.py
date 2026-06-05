import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# USER INPUTS
# =========================
MODEL_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\yolov8_obb_powerlines2\weights\best.pt"

# Replace with your two phone images (shifted 5–10cm sideways)
IMG1_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\cam\IMG-20251112-WA0045.jpg"
IMG2_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\cam\IMG-20251112-WA0028.jpg"
# Baseline distance between shots (in meters)
BASELINE = 0.08  # 8cm between pictures

# Focal length in pixels (approx for typical phone camera 26mm equivalent)
# We will refine this later using calibration
FOCAL_PIXELS = 1000


# =========================
# FUNCTIONS
# =========================
def get_wire_midpoint(results):
    """Extract midpoint of detected wire from YOLO OBB output"""

    if not results or not hasattr(results[0], "obb"):
        return None

    r = results[0]
    if not hasattr(r.obb, "xyxyxyxy"):
        return None

    quads = r.obb.xyxyxyxy.cpu().numpy()

    if len(quads) == 0:
        return None

    # Take only the first wire (best detection)
    quad = quads[0].reshape(4, 2)

    # Midpoint of quadrilateral (mean of 4 corner points)
    mid = quad.mean(axis=0)
    return mid  # (x, y)


def estimate_depth(f, baseline, disparity):
    """Depth = f * B / disparity"""
    if disparity < 1e-3:
        return None
    return (f * baseline) / disparity


# =========================
# MAIN LOGIC
# =========================
print("🔍 Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("📷 Reading images...")
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)

if img1 is None or img2 is None:
    print("⚠️ Error: Image paths invalid.")
    exit()

print("🤖 Detecting wire in image 1...")
res1 = model.predict(source=img1, imgsz=1024, conf=0.25, verbose=False)
mid1 = get_wire_midpoint(res1)

print("🤖 Detecting wire in image 2...")
res2 = model.predict(source=img2, imgsz=1024, conf=0.25, verbose=False)
mid2 = get_wire_midpoint(res2)

if mid1 is None or mid2 is None:
    print("⚠️ Wire not detected in one or both images.")
    exit()

x1, y1 = mid1
x2, y2 = mid2

disparity = abs(x1 - x2)

print(f"📌 Midpoint 1 = {mid1}")
print(f"📌 Midpoint 2 = {mid2}")
print(f"📏 Pixel Disparity = {disparity:.2f}px")

depth = estimate_depth(FOCAL_PIXELS, BASELINE, disparity)

if depth is None:
    print("⚠️ Disparity too small — cannot estimate depth")
else:
    print(f"✅ Estimated distance to wire = {depth:.2f} meters")

# Show detected images
cv2.circle(img1, (int(x1), int(y1)), 8, (0, 255, 0), -1)
cv2.circle(img2, (int(x2), int(y2)), 8, (0, 255, 0), -1)
cv2.imshow("Image 1 wire point", img1)
cv2.imshow("Image 2 wire point", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
