import os

label_folder = r"datasets/merged_yolo/labels/train"
bad_labels = []

for f in os.listdir(label_folder):
    if f.endswith(".txt"):
        path = os.path.join(label_folder, f)
        with open(path, "r") as file:
            lines = file.readlines()
            if len(lines) == 0:
                bad_labels.append(path)  # empty file
            else:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:  # invalid YOLO format
                        bad_labels.append(path)
                        break

print("Found", len(bad_labels), "bad label files")

for b in bad_labels[:20]:
    print(b)
