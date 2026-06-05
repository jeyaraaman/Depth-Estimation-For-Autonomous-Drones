import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model_path = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\detect\train_wire_gpu3\weights\best.pt"
model = YOLO(model_path)
 # use your trained weights here

# Input video path
video_path = "pic1.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 384))

    # ---------- YOLO DETECTION ----------
    results = model.predict(frame, conf=0.25, verbose=False)
    detections = results[0].boxes.data if len(results) else []

    # Draw YOLO detections
    if len(detections):
        for box in detections:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Wire {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---------- HOUGH LINE DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ---------- DISPLAY ----------
    cv2.imshow("Wire Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
