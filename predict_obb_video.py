from ultralytics import YOLO
import cv2
import numpy as np

# ✅ Load trained model
model = YOLO(r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt")

# ✅ Input video path
video_path = r"footage5.mp4"     # Change to your video file
cap = cv2.VideoCapture(video_path)

# ✅ Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_obb.mp4", fourcc, fps, (width, height))

# ✅ Tile size & overlap settings
tile = 1024
overlap = 200

def draw_obb(frame, obb_points):
    pts = obb_points.reshape(-1,2).astype(int)
    cv2.polylines(frame, [pts], True, (0,255,0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = frame.copy()

    h, w = frame.shape[:2]

    # 🔄 Sliding window tiles
    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            tile_img = frame[y:min(y + tile, h), x:min(x + tile, w)]

            preds = model(tile_img, conf=0.30, imgsz=tile, verbose=False)

            for r in preds:
                if r.obb is not None:
                    for poly in r.obb.xyxyxyxy.cpu().numpy():
                        draw_obb(result_frame, poly)

    out.write(result_frame)
    cv2.imshow("OBB Powerline Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Saved: output_obb.mp4")
