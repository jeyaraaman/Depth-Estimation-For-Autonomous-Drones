import os
import shutil
from tqdm import tqdm

# ====== CONFIGURATION ======
BASE_PATH = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets"

# We'll auto-detect the right outdoor folder
OUTDOOR_FOLDER_CANDIDATES = ["OBB_Data", "OBB Data"]
INDOOR_PATH = os.path.join(BASE_PATH, "indoor_obb")
MERGED_PATH = os.path.join(BASE_PATH, "merged_obb")

# Detect correct outdoor dataset path
OUTDOOR_PATH = None
for candidate in OUTDOOR_FOLDER_CANDIDATES:
    candidate_path = os.path.join(BASE_PATH, candidate)
    if os.path.exists(os.path.join(candidate_path, "train")):
        OUTDOOR_PATH = candidate_path
        break

if OUTDOOR_PATH is None:
    raise FileNotFoundError("❌ Could not find outdoor dataset folder (OBB_Data or OBB Data).")

print(f"✅ Found outdoor dataset at: {OUTDOOR_PATH}")

# Detect whether to use 'val' or 'valid'
if os.path.exists(os.path.join(OUTDOOR_PATH, "valid")):
    sets = ["train", "valid"]
else:
    sets = ["train", "val"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

print("🧩 Merging OBB Datasets...")
for s in sets:
    for sub in ["images", "labels"]:
        src_out = os.path.join(OUTDOOR_PATH, s, sub)
        src_in  = os.path.join(INDOOR_PATH, s, sub)
        dst     = os.path.join(MERGED_PATH, s, sub)
        ensure_dir(dst)

        count = 0
        for folder in [src_out, src_in]:
            if os.path.exists(folder):
                for f in tqdm(os.listdir(folder), desc=f"Merging {s}/{sub} from {os.path.basename(folder)}"):
                    src_file = os.path.join(folder, f)
                    dst_file = os.path.join(dst, f)
                    shutil.copy2(src_file, dst_file)
                    count += 1
        print(f"✅ Merged {count} files into {s}/{sub}")

print("✅ Merge complete!")

# Write data.yaml
yaml_path = os.path.join(MERGED_PATH, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""
path: "{MERGED_PATH.replace(os.sep, '/')}"
train: train/images
val: {sets[1]}/images
task: obb
nc: 1
names: ["cable"]
""")

print(f"📝 Created data.yaml at: {yaml_path}")
print("🚀 You can now train your universal model using this dataset.")
