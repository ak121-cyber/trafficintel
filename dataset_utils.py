from pathlib import Path
import yaml

def find_yolo_dataset(root):
    root = Path(root)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "images" / "train").exists() and (p / "labels" / "train").exists():
            return p
    raise FileNotFoundError(
        f"YOLO dataset not found under {root}. "
        "Expected images/train and labels/train."
    )

def write_yaml(dataset_root, names, output):
    dataset_root = Path(dataset_root).resolve()
    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": len(names),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output

def validate_dataset(dataset_root, class_count):
    dataset_root = Path(dataset_root)
    counts = {i: 0 for i in range(class_count)}
    errors = []

    for split in ("train", "val"):
        folder = dataset_root / "labels" / split
        if not folder.exists():
            errors.append(f"Missing {folder}")
            continue

        for file in folder.glob("*.txt"):
            for line_no, line in enumerate(file.read_text(errors="ignore").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    errors.append(f"{file}:{line_no}: expected 5 YOLO fields")
                    continue
                try:
                    cls = int(parts[0])
                    values = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"{file}:{line_no}: invalid label")
                    continue
                if cls not in counts:
                    errors.append(f"{file}:{line_no}: class {cls} outside range")
                else:
                    counts[cls] += 1
                if not all(0 <= x <= 1 for x in values):
                    errors.append(f"{file}:{line_no}: coordinates outside 0..1")
    return counts, errors
