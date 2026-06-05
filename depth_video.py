import cv2
import numpy as np
from ultralytics import YOLO
import os

# ================================
# CONFIG
# ================================
MODEL_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt"
VIDEO_SOURCE = r"test\f5.mp4"
RIGHT_REFERENCE_IMAGE = r"test/pic8R.jpg"

BASELINE = 0.10  # meters
FOCAL_PX = 7500  # calibrated focal px
GUIDE_MARGIN = 80
DANGER_ZONE = 0.30

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "result.mp4")


# ================================
# DETECT BEST OBB
# ================================
def detect_wire_center(model, img):
    try:
        results = model.predict(img, conf=0.15, verbose=False)
    except:
        return None

    r = results[0]
    if not hasattr(r, "obb") or len(r.obb) == 0:
        return None

    try:
        quads = r.obb.xyxyxyxy.cpu().numpy()
        confs = r.obb.conf.cpu().numpy()
    except:
        return None

    best = int(np.argmax(confs))
    quad = quads[best].reshape(4, 2)

    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None

    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))
    return (cx, cy), quad


# ================================
# DEPTH
# ================================
def compute_depth(cx_left, cx_right):
    disparity = abs(cx_left - cx_right)
    if disparity < 1:
        return None, disparity
    depth = (BASELINE * FOCAL_PX) / disparity
    return depth, disparity


# ================================
# MAIN
# ================================
def main():

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Loading right reference image...")
    right_img = cv2.imread(RIGHT_REFERENCE_IMAGE)
    if right_img is None:
        print("ERROR: Right reference image missing.")
        return

    right_det = detect_wire_center(model, right_img)
    if right_det is None:
        print("ERROR: No wire found in right reference image.")
        return

    (cx_right, cy_right), _ = right_det
    print(f"Right X-center: {cx_right:.2f}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    # -----------------------------
    # SETUP VIDEO WRITER
    # -----------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Saving output to {OUTPUT_VIDEO} ...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        frame_center_x = w // 2

        det = detect_wire_center(model, frame)
        if det is None:
            cv2.putText(frame, "Wire not found", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)
            cv2.imshow("WIRE", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        (cx_left, cy_left), quad = det

        depth, disp = compute_depth(cx_left, cx_right)

        # Draw wire box
        cv2.polylines(frame, [quad.astype(int)], True, (0, 255, 255), 2)
        cv2.circle(frame, (int(cx_left), int(cy_left)), 6, (0, 0, 255), -1)

        # Guidance
        if cx_left < frame_center_x - GUIDE_MARGIN:
            guide = "MOVE LEFT"
        elif cx_left > frame_center_x + GUIDE_MARGIN:
            guide = "MOVE RIGHT"
        else:
            guide = "HOLD POSITION"

        # Danger zone (lower part of frame)
        frac = cy_left / h
        if frac > 1 - DANGER_ZONE:
            danger = "DANGER!!"
            color = (0, 0, 255)
        else:
            danger = "SAFE"
            color = (0, 255, 0)

        # Distance text
        if depth is None:
            dtext = "Depth: TOO FAR"
        else:
            dtext = f"Depth: {depth:.2f} m"

        cv2.putText(frame, dtext, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, guide, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, danger, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # ----------------------------
        # SAVE TO OUTPUT VIDEO
        # ----------------------------
        out.write(frame)

        # Show
        cv2.imshow("WIRE", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n✅ Video saved successfully at:")
    print(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()