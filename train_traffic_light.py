from pathlib import Path
from ultralytics import YOLO
import torch

BASE = Path(__file__).resolve().parent
DATASET_ROOT = BASE / "datasets" / "traffic_light"
OUTPUT = BASE / "models" / "traffic_light"

def find_data_yaml():
    files = list(DATASET_ROOT.rglob("data.yaml"))

    if not files:
        raise FileNotFoundError(
            f"No data.yaml found inside {DATASET_ROOT}"
        )

    if len(files) > 1:
        print("Multiple data.yaml files found:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")

        choice = int(input("Select dataset number: ")) - 1
        return files[choice]

    return files[0]

def main():
    data_yaml = find_data_yaml()

    print(f"Dataset: {data_yaml}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model = YOLO("yolo26m.pt")

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=3,
        device=device,
        workers=4,
        project=str(OUTPUT),
        name="yolo26m",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=15,
        cache=True,
        amp=True,
        plots=True,
        save=True,
        val=True
    )

if __name__ == "__main__":
    main()