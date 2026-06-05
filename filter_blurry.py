import cv2
import os
import shutil

img_dir = r"datasets/merged_yolo/images/train"
label_dir = r"datasets/merged_yolo/labels/train"
bad_dir = r"datasets/merged_yolo/bad_images"

os.makedirs(bad_dir, exist_ok=True)

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    score = variance_of_laplacian(img)

    if score < 60:  # threshold (change to 50 if too strict)
        print(f"BLUR FOUND ({score:.2f}): {img_name}")

        txt_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")

        # move image
        shutil.move(img_path, os.path.join(bad_dir, img_name))

        # move label if exists
        label_path = os.path.join(label_dir, txt_name)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(bad_dir, txt_name))
