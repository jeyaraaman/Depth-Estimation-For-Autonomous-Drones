from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    model = YOLO("yolov8m.pt")  # fresh YOLO8 model

    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=50,
        imgsz=1024,
        batch=4,
        device=0,
        lr0=0.001,
        optimizer="AdamW",
        amp=True,
        workers=2,
        name="thin_stage1"
    )

if __name__ == "__main__":
    main()
