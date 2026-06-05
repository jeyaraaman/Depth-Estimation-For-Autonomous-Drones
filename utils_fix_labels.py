# utils_fix_labels.py
import os
from pathlib import Path

ROOT = Path("C:/Users/jeyar/wire_detection_project/yolo_gpu_env/datasets/merged_yolo")
lbl_train = ROOT/"labels"/"train"
lbl_val   = ROOT/"labels"/"val"

NC = 2  # set to number of classes you declared in data.yaml

def stats_and_fix(lbl_dir):
    total = 0
    class_counts = {}
    bad_files = []
    for f in lbl_dir.glob("*.txt"):
        total += 1
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        new_lines = []
        for line in lines:
            parts = line.split()
            cls = int(float(parts[0]))
            if cls >= NC:
                bad_files.append((f, line))
            # If label has 5 parts (YOLO) assume cls cx cy w h
            # If has 6 (OBB) treat first five same, angle is last
            if len(parts) >= 5:
                cx,cy,w,h = map(float, parts[1:5])
                # Ensure bbox sizes >0 and within [0,1]
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w  = min(max(w, 0.00001), 1.0)
                h  = min(max(h, 0.00001), 1.0)
                # If h or w negative (rare) fix by abs
                w = abs(w); h = abs(h)
                # Rebuild line (preserve extras like angle if present)
                rest = ""
                if len(parts) > 5:
                    rest = " " + " ".join(parts[5:])
                new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}{rest}")
            else:
                # keep as is
                new_lines.append(line)
            class_counts[cls] = class_counts.get(cls,0) + 1
        f.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return total, class_counts, bad_files

print("Checking train labels...")
ttrain, train_counts, train_bad = stats_and_fix(lbl_train)
print("train files:", ttrain, "class counts:", train_counts, "bad examples:", len(train_bad))
if train_bad:
    print("Sample bad train:", train_bad[:5])

print("Checking val labels...")
tval, val_counts, val_bad = stats_and_fix(lbl_val)
print("val files:", tval, "class counts:", val_counts, "bad examples:", len(val_bad))
if val_bad:
    print("Sample bad val:", val_bad[:5])

print("Done. If there are bad files, inspect them and fix class indices or set NC in data.yaml appropriately.")
