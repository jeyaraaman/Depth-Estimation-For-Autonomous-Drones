from ultralytics import YOLO
import cv2
import numpy as np
import torch

# ✅ Load your trained OBB model
model = YOLO(r"runs/obb/yolov8_obb_powerlines2/weights/best.pt")

# ✅ Open webcam (0 = default cam). Change to file path for USB camera
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("live_output.mp4", fourcc, 20.0, (640,480))


tile = 1024
overlap = 200


def draw_obb(frame, obb_points):
    pts = obb_points.reshape(-1, 2).astype(int)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    output = frame.copy()

    # 💡 Sliding tile inference for thin wires
    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            tile_img = frame[y:min(y + tile, h), x:min(x + tile, w)]

            preds = model(tile_img, conf=0.30, imgsz=tile, verbose=False)

            for r in preds:
                if r.obb is not None:
                    for poly in r.obb.xyxyxyxy.cpu().numpy():
                        draw_obb(output, poly)
    out.write(output)
    cv2.imshow("Live Powerline Detection (OBB)", output)

    # Quit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
