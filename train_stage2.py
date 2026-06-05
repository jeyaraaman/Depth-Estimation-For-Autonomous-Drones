from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    # ✅ Load best weights from Stage-1 fine-tune
    model = YOLO(r"runs/detect/train_yolov8m_finetune_stage1/weights/best.pt")

    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=30,                  # short boost training
        imgsz=896,                  # slightly higher resolution
        batch=4,                    # safe for 8GB VRAM
        device=0,
        amp=True,                   # mixed precision
        lr0=0.0003,                 # even smaller learning rate for stability
        optimizer="SGD",
        augment=True,
        workers=2,
        patience=10,                # stop early if no gains
        name="train_yolov8m_finetune_stage2",
        resume=False                # must stay false
    )

if __name__ == "__main__":
    main()
