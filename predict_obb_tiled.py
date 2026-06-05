from ultralytics import YOLO
import os
import cv2

# ✅ Load your trained model
model = YOLO(r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\obb\final_merged_train3\weights\best.pt")

# ✅ Folder containing test images
input_folder = r"test"
output_folder = r"tiled_results4"
os.makedirs(output_folder, exist_ok=True)

# Tile size & overlap
tile_size = 1024
overlap = 200

def tile_and_predict(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    results_img = img.copy()

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            r = model.predict(tile, imgsz=tile_size, conf=0.25, verbose=False)

            for result in r:
                if result.obb is not None:
                    for obb in result.obb.xyxyxyxy:
                        pts = obb.cpu().numpy().reshape(-1,2)
                        cv2.polylines(results_img, [pts.astype(int)], True, (0,255,0), 2)

    return results_img

# Run on all test images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        output_img = tile_and_predict(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, output_img)
        print(f"✅ Saved: {save_path}")
