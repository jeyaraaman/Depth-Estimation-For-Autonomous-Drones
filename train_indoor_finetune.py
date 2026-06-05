from ultralytics import YOLO
import torch

def main():
    # ✅ free up GPU memory
    torch.cuda.empty_cache()

    # =========================
    # MODEL SETUP
    # =========================
    # load your previous outdoor wire model (PL-V1)
    model = YOLO(r"runs/obb/yolov8_obb_powerlines2/weights/best.pt")

    # =========================
    # TRAINING CONFIGURATION
    # =========================
    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\indoor_cable_OBB\data.yaml",  # ✅ update to your local path
        epochs=30,                 # 20–40 is enough for fine-tune
        imgsz=1024,                # large image size helps thin cables
        batch=4,                   # adjust based on VRAM
        lr0=0.0005,                # smaller learning rate for fine-tuning
        freeze=10,                 # freeze backbone, train head
        patience=10,               # early stop if no improvement
        optimizer="SGD",           # stable for small datasets
        device=0,                  # GPU index
        workers=2,                 # reduce for laptops
        amp=True,                  # mixed precision = faster
        project="runs/obb",
        name="indoor_finetune",
        exist_ok=False,            # create a new run folder
        pretrained=True,           # continue from existing weights
        verbose=True
    )

    print("✅ Fine-tuning complete! Check results in runs/obb/indoor_finetune")

if __name__ == "__main__":
    main()
