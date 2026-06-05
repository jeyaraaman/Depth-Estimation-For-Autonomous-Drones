import os
import shutil

label_dir = r"datasets/merged_yolo/labels/train"
img_dir = r"datasets/merged_yolo/images/train"
save_dir = r"datasets/merged_yolo/empty_images"

os.makedirs(save_dir, exist_ok=True)

removed = 0
for label in os.listdir(label_dir):
    if label.endswith(".txt"):
        label_path = os.path.join(label_dir, label)

        # Check if file is empty (means no wires)
        if os.path.getsize(label_path) == 0:
            img_file = label.replace(".txt", ".jpg")
            img_file2 = label.replace(".txt", ".png")

            # Move label
            shutil.move(label_path, os.path.join(save_dir, label))

            # Move image if exists
            for img in [img_file, img_file2]:
                img_path = os.path.join(img_dir, img)
                if os.path.exists(img_path):
                    shutil.move(img_path, os.path.join(save_dir, img))

            removed += 1

print(f"✅ Moved {removed} empty-label images")
