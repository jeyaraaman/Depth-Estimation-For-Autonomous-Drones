from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()
    model = YOLO(r"runs/detect/train_yolov8m_100ep/weights/last.pt")
    #model.freeze(10)  # freezes first 10 layers

    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=75,
        imgsz=832,
        batch=8,
        device=0,
        amp=True,
        lr0=0.0005,
        optimizer="SGD",
        augment=True,
        workers=2,
        name="train_yolov8m_finetune_stage1",
        resume=False
    )

if __name__ == "__main__":
    main()
