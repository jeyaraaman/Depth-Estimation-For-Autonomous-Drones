import os
from PIL import Image

folder = r"datasets/merged_yolo/images/val"
bad = []

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    try:
        img = Image.open(path)
        w,h = img.size
        if w < 600 or h < 600:
            bad.append(path)
    except:
        bad.append(path)

print("Found", len(bad), "low-quality files")
for b in bad:
    print(b)
