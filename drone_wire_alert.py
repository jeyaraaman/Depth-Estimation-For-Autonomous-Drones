import cv2
import numpy as np
from ultralytics import YOLO
import winsound  # warning beep

# ---------------- SETTINGS ----------------
MODEL_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt"
VIDEO_SOURCE = r"test\f5.mp4"  # webcam / or path to video.mp4
DANGER_ZONE = 0.30
BEEP = True
# ------------------------------------------

def play_warning():
    try:
        if BEEP:
            winsound.Beep(1000, 150)
    except:
        pass

def get_direction(frame_center_x, wire_center_x):
    if wire_center_x < frame_center_x - 80:
        return "⬅ MOVE LEFT"
    elif wire_center_x > frame_center_x + 80:
        return "➡ MOVE RIGHT"
    else:
        return "⬆ MOVE UP / STOP"

def detect_wires(model, frame):
    results = model(frame, verbose=False)
    h, w = frame.shape[:2]
    frame_center_x = w // 2
    guidance_text = "✅ CLEAR"
    danger = False

    if not hasattr(results[0], "obb"):
        cv2.putText(frame, "⚠ No OBB output!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return frame

    for r in results:
        obb_data = r.obb.xywhr  # ✅ FIXED here

        if obb_data is None:
            continue

        for b in obb_data.cpu().numpy():
            cx, cy, w_box, h_box, angle = b

            # angle may be radians or degrees; normalize
            if abs(angle) < np.pi:  # radians → degrees
                angle_deg = angle * 180 / np.pi
            else:
                angle_deg = angle

            # convert rotated rect -> corner points
            rect = ((cx, cy), (w_box, h_box), angle_deg)
            pts = cv2.boxPoints(rect).astype(int)

            xs = pts[:, 0]
            ys = pts[:, 1]

            wire_center_y = np.mean(ys)
            wire_center_x = np.mean(xs)
            wire_fraction = wire_center_y / h

            if wire_fraction > (1 - DANGER_ZONE):
                color = (0, 0, 255)
                danger = True
                guidance_text = get_direction(frame_center_x, wire_center_x)
                play_warning()
            elif wire_fraction > 0.55:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.polylines(frame, [pts], True, color, 2)

    cv2.putText(frame, guidance_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0,0,255) if danger else (0,255,0), 3)

    if danger:
        cv2.putText(frame, "⚠ WIRE AHEAD!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    return frame


def main():
    print("🚁 Drone Wire Warning System Starting...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌ camera/video error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_wires(model, frame)
        cv2.imshow("WIRE ALERT SYSTEM", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
