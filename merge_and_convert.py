# merge_and_convert.py
import os, math, shutil
from pathlib import Path
from PIL import Image

ROOT = Path(r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env")
A = ROOT/"datasets"/"data_A"
B = ROOT/"datasets"/"data_B"
OUT = ROOT/"datasets"/"merged_yolo"   # axis-aligned YOLO output

for p in ["images/train","images/val","labels/train","labels/val"]:
    (OUT/p).mkdir(parents=True, exist_ok=True)

def is_obb_line(parts):
    # heuristics: OBB may have 6 values per line: cls cx cy w h angle
    return len(parts) == 6

def obb_to_aabb(cx, cy, w, h, angle_deg, img_w, img_h):
    # convert normalized (cx,cy,w,h) + angle (deg) to axis-aligned bbox normalized
    # Steps: compute corner points, rotate, compute AABB, normalize
    angle = math.radians(angle_deg)
    # convert normalized to pixels
    cx_p, cy_p, w_p, h_p = cx*img_w, cy*img_h, w*img_w, h*img_h
    # four corners centered at origin before rotation
    dx = w_p/2; dy = h_p/2
    corners = [(-dx,-dy), (dx,-dy), (dx,dy), (-dx,dy)]
    # rotate and translate
    pts = []
    for (x,y) in corners:
        xr = x*math.cos(angle) - y*math.sin(angle) + cx_p
        yr = x*math.sin(angle) + y*math.cos(angle) + cy_p
        pts.append((xr,yr))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = max(0,min(xs)), min(img_w,max(xs))
    y_min, y_max = max(0,min(ys)), min(img_h,max(ys))
    a_w = max(1.0, x_max - x_min)
    a_h = max(1.0, y_max - y_min)
    a_cx = (x_min + x_max)/2
    a_cy = (y_min + y_max)/2
    # normalize
    return a_cx/img_w, a_cy/img_h, a_w/img_w, a_h/img_h

def process_dataset(src):
    for split in ["train","val"]:
        img_src = Path(src)/"images"/split
        lbl_src = Path(src)/"labels"/split
        img_out = OUT/"images"/split
        lbl_out = OUT/"labels"/split
        if not img_src.exists():
            continue
        for img_path in img_src.iterdir():
            if not img_path.is_file(): continue
            # copy image (ensure jpg extension)
            target_img_name = img_path.stem + img_path.suffix
            shutil.copy2(img_path, img_out/target_img_name)
            # process label
            lbl_file = lbl_src/(img_path.stem + ".txt")
            out_lbl = lbl_out/(img_path.stem + ".txt")
            lines_out = []
            if lbl_file.exists():
                with open(lbl_file,"r") as f:
                    for line in f.read().splitlines():
                        if not line.strip(): continue
                        parts = line.strip().split()
                        if is_obb_line(parts):
                            # class cx cy w h angle
                            cls = int(parts[0])
                            cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4]); ang = float(parts[5])
                            # get image size
                            from PIL import Image
                            img = Image.open(img_path)
                            iw, ih = img.size
                            nx, ny, nw, nh = obb_to_aabb(cx, cy, w, h, ang, iw, ih)
                            # clamp to 0..1
                            nx = min(max(nx,0.0),1.0); ny = min(max(ny,0.0),1.0)
                            nw = min(max(nw,0.0),1.0); nh = min(max(nh,0.0),1.0)
                            lines_out.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                        else:
                            # assume already YOLO normal: cls cx cy w h
                            cls = parts[0]; cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # write label (may be empty)
            with open(out_lbl,"w") as f:
                for L in lines_out:
                    f.write(L+"\n")

# process both datasets
process_dataset(A); process_dataset(B)
print("Merged (and converted if needed) into", OUT)
