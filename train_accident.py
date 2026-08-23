from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
from dataset_utils import find_yolo_dataset, write_yaml, validate_dataset
from config import *

def main():
    dataset = find_yolo_dataset(ACCIDENT_DATASET)
    counts, errors = validate_dataset(dataset, len(ACCIDENT_NAMES))
    print("Accident dataset:", dataset)
    print("Class counts:", counts)

    if errors:
        print("\nDataset errors:")
        for e in errors[:20]:
            print(e)
        raise RuntimeError("Fix the dataset before training.")

    yaml_path = ROOT / "datasets" / "accident.yaml"
    write_yaml(dataset, ACCIDENT_NAMES, yaml_path)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_MODEL)

    model.train(
        data=str(yaml_path),
        epochs=60,
        imgsz=640,
        batch=1,
        device=device,
        workers=0,
        pretrained=True,
        amp=torch.cuda.is_available(),
        cache=False,
        patience=15,
        close_mosaic=10,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=3.0,
        translate=0.05,
        scale=0.20,
        shear=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.25,
        project=str(ROOT / "models" / "accident"),
        name="yolo26m",
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )

    best = ROOT / "models" / "accident" / "yolo26m" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    final = ROOT / "models" / "accident" / "best.pt"
    shutil.copy2(best, final)
    print("Saved:", final)

if __name__ == "__main__":
    main()
