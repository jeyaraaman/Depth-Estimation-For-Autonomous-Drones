# thicken_wires.py
import cv2, os

DATA_DIR = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo"
IMAGE_DIRS = ["images/train", "images/val"]
OUT_PREFIX = "_thick"  # appended to filename before extension
kernel_size = 3  # dilation kernel (3 or 5)

def process_image(in_path, out_path, kernel_size=3):
    img = cv2.imread(in_path)
    if img is None: return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect edges to target wires, then dilate
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)  # adjust threshold if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dil = cv2.dilate(th, kernel, iterations=1)
    # blend dilated mask to original to thicken bright lines
    mask = dil.astype(bool)
    out = img.copy()
    out[mask] = cv2.addWeighted(out, 0.5, out, 0.5, 50)[mask]  # subtle boost
    cv2.imwrite(out_path, out)
    return True

def main():
    for rel in IMAGE_DIRS:
        img_dir = os.path.join(DATA_DIR, rel)
        out_dir = os.path.join(DATA_DIR, rel + OUT_PREFIX)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(img_dir):
            if not f.lower().endswith((".jpg",".jpeg",".png")): continue
            in_path = os.path.join(img_dir, f)
            name, ext = os.path.splitext(f)
            out_path = os.path.join(out_dir, name + ext)
            process_image(in_path, out_path, kernel_size)
    print("Done creating thickened images. Saved to *_thick folders.")

if __name__ == "__main__":
    main()
