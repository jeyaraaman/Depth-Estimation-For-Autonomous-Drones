from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    # Load OBB model (start with yolov8m-obb)
    model = YOLO("yolov8m-obb.pt")

    model.train(
        data=r"C:/Users/jeyar/wire_detection_project/yolo_gpu_env/datasets/OBB Data/data.yaml",
        epochs=100,
        imgsz=1024,
        batch=2,         # change to 4 if VRAM allows
        device=0,
        workers=2,
        amp=True,
        name="yolov8_obb_powerlines"
    )

if __name__ == "__main__":
    main()
