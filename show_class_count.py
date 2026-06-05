# show_class_counts.py
from pathlib import Path
from collections import Counter
p = Path("C:/Users/jeyar/wire_detection_project/yolo_gpu_env/datasets/merged_yolo/labels/val")
cnt = Counter()
for f in p.glob("*.txt"):
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        cls = int(float(line.split()[0]))
        cnt[cls]+=1
print("class counts in VAL:", dict(sorted(cnt.items())))
