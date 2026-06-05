from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    # ✅ Use the best Stage-2 weights
    model = YOLO(r"runs/detect/train_yolov8m_finetune_stage2/weights/best.pt")

    # ✅ Freeze backbone — train only detection head
   # model.freeze(20)

    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=15,              # short final tuning
        imgsz=1024,             # highest clarity for wires
        batch=2,                # fits 8GB VRAM
        lr0=0.0002,             # very small lr
        device=0,
        amp=True,
        optimizer="SGD",
        workers=2,
        name="train_yolov8m_polish_stage3",
        resume=False
    )

if __name__ == "__main__":
    main()
