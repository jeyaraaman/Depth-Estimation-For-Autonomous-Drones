from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    # ✅ Load thick+thin merged model
    model = YOLO(r"runs/detect/thick_stage12/weights/best.pt")

    # ✅ Manually freeze first 10 layers
    for name, param in model.model.named_parameters():
        # Freeze backbone layers only
        if any(layer in name for layer in ["0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]):
            param.requires_grad = False

    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=15,
        imgsz=1280,       # 🔥 high resolution
        batch=2,
        device=0,
        lr0=0.00025,
        optimizer="AdamW",
        amp=True,
        workers=2,
        name="final_polish_stage3",
        resume=False
    )

if __name__ == "__main__":
    main()
