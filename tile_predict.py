from ultralytics import YOLO
import cv2
import os
import math

model = YOLO(r"runs/detect/final_polish_stage3/weights/best.pt")

source_folder = "test_images"
output_folder = "tiled_output"
os.makedirs(output_folder, exist_ok=True)

tile_size = 1024
overlap = 0.25   # overlap % to avoid cut wires

for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    step = int(tile_size * (1 - overlap))
    results_combined = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.size == 0:
                continue

            result = model.predict(tile, imgsz=tile_size, conf=0.20, verbose=False)
            for r in result:
                if r.boxes is not None:
                    for b in r.boxes.xyxy.cpu().numpy():
                        bx1, by1, bx2, by2 = map(int, b)
                        # Shift coords back to global image
                        results_combined.append([bx1+x, by1+y, bx2+x, by2+y])

    # Draw boxes
    for (x1, y1, x2, y2) in results_combined:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imwrite(os.path.join(output_folder, img_name), img)

print("✅ Tiled inference completed. Check 'tiled_output/' folder.")
