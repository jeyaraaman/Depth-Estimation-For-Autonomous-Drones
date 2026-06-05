import cv2
import os
import numpy as np
from ultralytics import YOLO

# -----------------------------
# YOUR INPUT IMAGES
# -----------------------------
LEFT_IMAGE = r"test/pic8L.jpg"
RIGHT_IMAGE = r"test/pic8R.jpg"

# Your OBB wire model
MODEL_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt"

# -----------------------------
# CALIBRATED CAMERA VALUES
# -----------------------------
BASELINE = 0.1  # meters
FOCAL_PX = 7500  # from calibration


# ==================================
# DETECT WIRE CENTER USING YOLO OBB
# ==================================
def detect_wire_center(model, img):
    results = model.predict(img, conf=0.15, verbose=False)
    r = results[0]

    if not hasattr(r, "obb") or len(r.obb) == 0:
        return None

    quads = r.obb.xyxyxyxy.cpu().numpy()
    confs = r.obb.conf.cpu().numpy()

    best = int(np.argmax(confs))
    quad = quads[best].reshape(4, 2)

    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))

    return (cx, cy), quad


# --------------------------
# DEPTH FROM DISPARITY
# --------------------------
def estimate_depth(cx_left, cx_right):
    disparity = abs(cx_left - cx_right)
    if disparity < 1:
        return None, disparity
    depth = (BASELINE * FOCAL_PX) / disparity
    return depth, disparity


# --------------------------
# CREATE 3D ANAGLYPH IMAGE
# --------------------------
def create_anaglyph(img_left, img_right):
    """Creates Red-Cyan 3D stereo image."""
    # Resize to same size
    h = min(img_left.shape[0], img_right.shape[0])
    w = min(img_left.shape[1], img_right.shape[1])

    img_left = cv2.resize(img_left, (w, h))
    img_right = cv2.resize(img_right, (w, h))

    # Split channels
    left_r, left_g, left_b = cv2.split(img_left)
    right_r, right_g, right_b = cv2.split(img_right)

    # Anaglyph (Red from Left, Cyan from Right)
    anaglyph = cv2.merge((left_r, right_g, right_b))

    return anaglyph


# --------------------------
# MAIN
# --------------------------
def main():

    os.makedirs("output", exist_ok=True)

    print("📦 Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("📷 Reading input images...")
    imgL = cv2.imread(LEFT_IMAGE)
    imgR = cv2.imread(RIGHT_IMAGE)

    if imgL is None or imgR is None:
        print("❌ ERROR: Could not read images")
        return

    print("🔎 Detecting wire in left image...")
    left = detect_wire_center(model, imgL)

    print("🔎 Detecting wire in right image...")
    right = detect_wire_center(model, imgR)

    if left is None or right is None:
        print("❌ Wire not detected in one or both images.")
        return

    (ptL, quadL) = left
    (ptR, quadR) = right

    # DEPTH CALCULATION
    depth, disparity = estimate_depth(ptL[0], ptR[0])

    print(f"\n📏 Disparity(px): {disparity:.2f}")
    if depth is None:
        print("⚠ Depth too small to compute.")
    else:
        print(f"📏 Estimated Distance: {depth:.2f} meters")

    # Draw detections
    imgL2 = imgL.copy()
    imgR2 = imgR.copy()

    cv2.polylines(imgL2, [quadL.astype(int)], True, (0, 255, 255), 2)
    cv2.polylines(imgR2, [quadR.astype(int)], True, (0, 255, 255), 2)

    cv2.circle(imgL2, (int(ptL[0]), int(ptL[1])), 6, (0, 0, 255), -1)
    cv2.circle(imgR2, (int(ptR[0]), int(ptR[1])), 6, (0, 0, 255), -1)

    # =========================================
    # ⭐ CREATE 3D ANAGLYPH IMAGE ⭐
    # =========================================
    anaglyph = create_anaglyph(imgL2, imgR2)
    cv2.imwrite("output/3d_anaglyph.jpg", anaglyph)

    print("\n🎉 3D Anaglyph Image Created!")
    print("   → output/3d_anaglyph.jpg")

    # Save normal outputs too
    cv2.imwrite("output/left_result.jpg", imgL2)
    cv2.imwrite("output/right_result.jpg", imgR2)
    cv2.imwrite("output/stitched.jpg", np.hstack((imgL2, imgR2)))

    # Display
    cv2.imshow("3D VIEW (Use Red-Cyan Glasses!)", anaglyph)
    cv2.imshow("Left", imgL2)
    cv2.imshow("Right", imgR2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()