from ultralytics import YOLO
import torch
import os

def main():
    torch.cuda.empty_cache()

    # ✅ Load THIN model weights
    model = YOLO(r"runs/detect/thin_stage1/weights/best.pt")

    # ✅ Replace training paths to use thick-wire folder only
    os.environ["YOLO_TRAIN_IMAGES"] = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\images\train_thick_filtered"
    os.environ["YOLO_TRAIN_LABELS"] = r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\labels\train_thick_filtered"

    # ✅ Now train thick wires only
    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=30,
        imgsz=1024,
        batch=2,
        device=0,
        lr0=0.0004,
        optimizer="AdamW",
        amp=True,
        workers=2,
        name="thick_stage1",
        resume=False
    )

if __name__ == "__main__":
    main()
