from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ERROR_COLORS = (
    ("Khớp", "#4ADE80"),
    ("Dự đoán thừa", "#FB923C"),
    ("Bỏ sót", "#38BDF8"),
)


def read_image(path: Path, flags: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def compare_mask(path: Path, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    prediction = read_image(path, cv2.IMREAD_GRAYSCALE) > 127
    height, width = prediction.shape
    reference = cv2.resize(target.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlap = np.count_nonzero(prediction & reference)
    area_sum = np.count_nonzero(prediction) + np.count_nonzero(reference)
    smooth = 1e-7
    dice = (2 * overlap + smooth) / (area_sum + smooth)
    iou = (overlap + smooth) / (area_sum - overlap + smooth)
    return prediction, reference, dice, iou


def show_overlay(axis: plt.Axes, image: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    height, width = prediction.shape
    axis.imshow(cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR))
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    regions = (prediction & target, prediction & ~target, ~prediction & target)
    for region, (_, color) in zip(regions, ERROR_COLORS):
        overlay[region] = matplotlib.colors.to_rgba(color, alpha=0.65)
    axis.imshow(overlay, interpolation="nearest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved Kvasir-SEG masks on the fixed test split.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing images/ and masks/.")
    parser.add_argument("--split-file", type=Path, default=Path("kvasir-seg.json"))
    parser.add_argument("--run", nargs=2, metavar=("LABEL", "PREDICTION_DIR"), action="append", required=True,
                        help="Repeat for each model. The first run determines which images are plotted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test-visualization"))
    args = parser.parse_args()

    runs = [(label, Path(directory)) for label, directory in args.run]
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        parser.error("Run labels must be unique.")

    names = json.loads(args.split_file.read_text(encoding="utf-8"))["test"]
    if len(names) < 3 or len({Path(name).stem for name in names}) != len(names):
        parser.error("The test split needs at least three images with unique filename stems.")

    scores = {}
    for name in names:
        target = read_image(args.dataset_root / "masks" / name, cv2.IMREAD_GRAYSCALE) > 127
        pred_name = f"{Path(name).stem}.png"
        scores[name] = [
            compare_mask(directory / pred_name, target)[2:]
            for _, directory in runs
        ]

    ranked = sorted(names, key=lambda name: (scores[name][0][1], name))
    count = min(2, len(ranked) // 3)
    middle = (len(ranked) - count) // 2
    groups = (
        ("Tốt", ranked[-count:][::-1]),
        ("Trung vị", ranked[middle:middle + count]),
        ("Kém", ranked[:count]),
    )
    selected = {name: group for group, images in groups for name in images}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per-image-scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        fields = [f"{label} {metric}" for label in labels for metric in ("Dice", "IoU")]
        writer.writerow(["image", "nhóm hiển thị", *fields])
        for name in ranked:
            writer.writerow([name, selected.get(name, ""), *[score for pair in scores[name] for score in pair]])

    num_rows = len(selected)
    num_columns = 2 + len(runs)
    fig, axes = plt.subplots(num_rows, num_columns, squeeze=False, figsize=(2.8 * num_columns, 2.35 * num_rows))
    fig.suptitle(f"Kvasir-SEG · Ảnh test theo IoU từng ảnh của {labels[0]}", fontsize=15)
    fig.legend(handles=[Patch(color=color, label=label) for label, color in ERROR_COLORS],
               loc="lower center", ncol=len(ERROR_COLORS), frameon=False)
    for row, (name, group) in enumerate(selected.items()):
        image = read_image(args.dataset_root / "images" / name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = read_image(args.dataset_root / "masks" / name, cv2.IMREAD_GRAYSCALE) > 127
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{group} · {Path(name).stem}", fontsize=8)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Mask chuẩn", fontsize=9)
        for col, (label, directory) in enumerate(runs, start=2):
            prediction, reference, dice, iou = compare_mask(directory / f"{Path(name).stem}.png", target)
            show_overlay(axes[row, col], image, prediction, reference)
            axes[row, col].set_title(f"{label} · Dice {dice:.3f} · IoU {iou:.3f}", fontsize=9)
        for axis in axes[row]:
            axis.set_axis_off()

    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(args.output_dir / "test-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Compared {len(names)} test images across {len(runs)} run(s). Output: {args.output_dir}")


if __name__ == "__main__":
    main()

