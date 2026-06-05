import os, shutil

label_dir = r"datasets/merged_yolo/labels/train"
img_dir = r"datasets/merged_yolo/images/train"

thick_dir_labels = r"datasets/merged_yolo/labels/train_thick_filtered"
thick_dir_images = r"datasets/merged_yolo/images/train_thick_filtered"

os.makedirs(thick_dir_labels, exist_ok=True)
os.makedirs(thick_dir_images, exist_ok=True)


def is_thick(label_file):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, _, _, _, h = parts
                if float(h) > 0.60:
                    return True
    return False


count = 0

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        lbl_path = os.path.join(label_dir, file)
        if is_thick(lbl_path):
            shutil.move(lbl_path, os.path.join(thick_dir_labels, file))

            img_base = file.replace(".txt", "")
            for ext in [".jpg", ".png"]:
                img_src = os.path.join(img_dir, img_base + ext)
                if os.path.exists(img_src):
                    shutil.move(img_src, os.path.join(thick_dir_images, img_base + ext))

            count += 1

print(f"✅ Done! Moved {count} thick-wire images.")
