from pathlib import Path

ROOT = Path("C:/Users/jeyar/wire_detection_project/yolo_gpu_env/datasets/merged_yolo/labels")

for split in ["train", "val"]:
    folder = ROOT / split
    for f in folder.glob("*.txt"):
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        out_lines = []
        changed = False
        for line in lines:
            parts = line.split()
            cls = int(float(parts[0]))
            if cls != 0:  # change all non-zero classes to 0
                parts[0] = "0"
                changed = True
            out_lines.append(" ".join(parts))
        if changed:
            f.write_text("\n".join(out_lines) + "\n")

print("✅ Remapping complete: All classes > 0 are now class 0 (for train & val).")
