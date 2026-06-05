import cv2
import os
import numpy as np
from ultralytics import YOLO

# -----------------------------
# INPUT IMAGES
# -----------------------------
LEFT_IMAGE = r"test/pic8L.jpg"
RIGHT_IMAGE = r"test/pic8R.jpg"

# Wire detection model
MODEL_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt"

# -----------------------------
# CAMERA PARAMETERS
# -----------------------------
BASELINE = 0.10   # 6 cm between cameras (measure accurately!)
REAL_DISTANCE = 4.5  # <--- REAL distance to wire (in meters)

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

    # center point of quad
    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))

    return (cx, cy), quad


# ==================================
# COMPUTE DEPTH FROM DISPARITY
# ==================================
def estimate_depth(cx_left, cx_right, focal_px):
    disparity = abs(cx_left - cx_right)
    if disparity < 1:
        return None, disparity
    depth = (BASELINE * focal_px) / disparity
    return depth, disparity


# ==================================
# COMPUTE FOCAL LENGTH (px)
# Using known real distance + disparity
# ==================================
def compute_focal_px(cx_left, cx_right, baseline, real_distance):
    disparity = abs(cx_left - cx_right)
    if disparity < 1:
        print("❌ Disparity too small for focal calculation.")
        return None
    focal_px = (real_distance * disparity) / baseline
    return focal_px


# ==================================
# MAIN
# ==================================
def main():

    os.makedirs("output", exist_ok=True)

    print("📦 Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("📷 Reading input images...")
    imgL = cv2.imread(LEFT_IMAGE)
    imgR = cv2.imread(RIGHT_IMAGE)

    if imgL is None or imgR is None:
        print("❌ ERROR: Could not read one or both images!")
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

    # -----------------------------------------
    # STEP 1: Compute disparity
    # -----------------------------------------
    disparity = abs(ptL[0] - ptR[0])
    print(f"\n📏 Disparity(px): {disparity:.2f}")

    # -----------------------------------------
    # STEP 2: Estimate FOCAL LENGTH from this stereo pair
    # -----------------------------------------
    focal_px = compute_focal_px(ptL[0], ptR[0], BASELINE, REAL_DISTANCE)

    if focal_px is None:
        return

    print(f"🎯 Estimated FOCAL LENGTH (pixels): {focal_px:.2f}")

    # -----------------------------------------
    # STEP 3: Compute depth using estimated focal length
    # -----------------------------------------
    depth, _ = estimate_depth(ptL[0], ptR[0], focal_px)

    print(f"📏 Estimated Depth using this focal: {depth:.2f} meters")

    # -----------------------------------------
    # Draw detections
    # -----------------------------------------
    imgL2 = imgL.copy()
    imgR2 = imgR.copy()

    cv2.polylines(imgL2, [quadL.astype(int)], True, (0, 255, 255), 2)
    cv2.polylines(imgR2, [quadR.astype(int)], True, (0, 255, 255), 2)

    cv2.circle(imgL2, (int(ptL[0]), int(ptL[1])), 6, (0, 0, 255), -1)
    cv2.circle(imgR2, (int(ptR[0]), int(ptR[1])), 6, (0, 0, 255), -1)

    # Save
    cv2.imwrite("output/left_result.jpg", imgL2)
    cv2.imwrite("output/right_result.jpg", imgR2)
    cv2.imwrite("output/stitched.jpg", np.hstack((imgL2, imgR2)))

    print("\n✅ Saved output images.")

    # Show results
    cv2.imshow("LEFT - Detection", imgL2)
    cv2.imshow("RIGHT - Detection", imgR2)
    cv2.imshow("Stitched", np.hstack((imgL2, imgR2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()