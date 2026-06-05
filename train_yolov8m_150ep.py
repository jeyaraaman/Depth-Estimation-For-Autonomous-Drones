from ultralytics import YOLO
import torch

def main():
    # 🔹 Clear any cached GPU memory from previous runs
    torch.cuda.empty_cache()

    # 🔹 Load the previously trained weights
    model = YOLO(r"runs/detect/train_yolov8m_100ep/weights/best.pt")

    # 🔹 Continue training with optimized stable settings
    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=150,                # continue to 150 epochs
        imgsz=960,                 # reduced slightly from 1280 to save VRAM, still good for wires
        batch=4,                   # small batch size for 8GB GPU
        workers=2,                 # limit dataloader threads (avoid Windows memory errors)
        device=0,                  # GPU device ID
        amp=True,                  # enable mixed precision for VRAM efficiency
        lr0=0.001,                 # smaller learning rate for fine-tuning
        optimizer="SGD",           # more stable for continued training
        patience=30,               # early stopping if no improvement
        augment=True,              # data augmentation
        hsv_h=0.015,               # slight hue shift
        hsv_s=0.7,                 # strong saturation variation
        hsv_v=0.4,                 # brightness variation
        degrees=5,                 # small rotation
        translate=0.2,             # small translation
        scale=0.5,                 # scale variation
        fliplr=0.5,                # horizontal flip
        name="train_yolov8m_150ep_continued",  # output folder name
        project="runs/detect"
    )

if __name__ == "__main__":
    main()
