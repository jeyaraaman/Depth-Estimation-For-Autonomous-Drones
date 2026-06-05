# expand_labels.py
import os
import cv2

# paths (change if needed)
DATA_DIR = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo"
LABEL_DIRS = ["labels/train", "labels/val"]  # paths relative to DATA_DIR
IMAGE_DIRS = ["images/train", "images/val"]  # corresponding image directories
pad_px = 10  # expand bbox by this many pixels on each side

def expand_one_label(lbl_path, img_w, img_h, pad_px):
    lines = []
    with open(lbl_path, "r") as f:
        for l in f.read().strip().splitlines():
            if not l.strip():
                continue
            parts = l.split()
            cls = parts[0]
            xc, yc, w, h = map(float, parts[1:5])
            # convert normalized to pixels
            x_c = xc * img_w
            y_c = yc * img_h
            bw = w * img_w
            bh = h * img_h
            x0 = x_c - bw/2
            y0 = y_c - bh/2
            x1 = x_c + bw/2
            y1 = y_c + bh/2
            # expand
            x0 = max(0, x0 - pad_px)
            y0 = max(0, y0 - pad_px)
            x1 = min(img_w, x1 + pad_px)
            y1 = min(img_h, y1 + pad_px)
            # convert back to normalized
            new_w = (x1 - x0) / img_w
            new_h = (y1 - y0) / img_h
            new_xc = (x0 + x1) / 2 / img_w
            new_yc = (y0 + y1) / 2 / img_h
            lines.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")
    return lines

def main():
    for lbl_rel, img_rel in zip(LABEL_DIRS, IMAGE_DIRS):
        lbl_dir = os.path.join(DATA_DIR, lbl_rel)
        img_dir = os.path.join(DATA_DIR, img_rel)
        if not os.path.isdir(lbl_dir):
            print("Label dir not found:", lbl_dir); continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"): continue
            lbl_path = os.path.join(lbl_dir, fname)
            # find corresponding image by name guess (common patterns used)
            basename = os.path.splitext(fname)[0]
            # search common extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                p = os.path.join(img_dir, basename + ext)
                if os.path.exists(p):
                    img_path = p; break
            if img_path is None:
                # try patterns where label filenames include .rf. etc
                # find file by in directory containing basename start
                matches = [f for f in os.listdir(img_dir) if f.startswith(basename.split("_")[0])]
                if matches:
                    img_path = os.path.join(img_dir, matches[0])
            if img_path is None:
                print("Image not found for label:", lbl_path)
                continue
            img = cv2.imread(img_path)
            if img is None:
                print("Failed to read image:", img_path); continue
            h, w = img.shape[:2]
            new_lines = expand_one_label(lbl_path, w, h, pad_px)
            # overwrite label (or write to new folder if you want backup)
            with open(lbl_path, "w") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print("Done expanding labels by", pad_px, "px")

if __name__ == "__main__":
    main()
