import os

label_dir = r"datasets/merged_yolo/labels/train"

def is_thick(label_file):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, _, _, w, h = parts
                w, h = float(w), float(h)
                # If bounding box covers >60% height → thick cable
                if h > 0.60:
                    return True
    return False

thick_files = []

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        if is_thick(os.path.join(label_dir, file)):
            thick_files.append(file)

print(f"Found {len(thick_files)} thick-wire label files:")
for f in thick_files:
    print(f)
