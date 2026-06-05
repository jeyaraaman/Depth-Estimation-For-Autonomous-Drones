from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLOv8 model
model = YOLO(r"runs/detect/train_yolov8m_finetune_stage1/weights/best.pt")

input_folder = "test_images"
output_folder = "tiled_results"
os.makedirs(output_folder, exist_ok=True)

tile_size = 1024
overlap = 256  # 25% overlap for small wires

def tile_image(img, tile_size, overlap):
    h, w, _ = img.shape
    tiles = []
    coords = []

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile = img[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            coords.append((x, y))
    return tiles, coords

for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(input_folder, file))
    tiles, coords = tile_image(img, tile_size, overlap)

    all_boxes = []
    for tile, (x0, y0) in zip(tiles, coords):
        results = model(tile, imgsz=tile_size, conf=0.15, iou=0.4)

        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                all_boxes.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0])

    # Draw boxes on original image
    for x1, y1, x2, y2 in all_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, file), img)

print("✅ Tiled inference complete — check tiled_results folder")
