import os

labels_dir = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\labels\train"

for file in os.listdir(labels_dir):
    path = os.path.join(labels_dir, file)
    with open(path, "r") as f:
        lines = f.readlines()
    fixed = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            parts[0] = "0"  # force class to 0 (wire)
            fixed.append(" ".join(parts) + "\n")
    with open(path, "w") as f:
        f.writelines(fixed)

print("✅ All label class IDs fixed to 0")
