"""
Run COCO-pretrained YOLOv8 on a sample of BDD100K images and save
annotated output images so you can visually check detection quality.

This uses NO fine-tuning — it is the raw pretrained model straight from
Ultralytics. The purpose is to see how well it already performs on
driving scenes before committing to training.

Usage:
    python scripts/baseline_check.py \
        --images data/bdd100k/images/100k/val \
        --output output/baseline \
        --n 20

    --n controls how many images to sample (default 20).
    Output images are saved to the --output folder with boxes drawn on them.
"""

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_COLOURS = {
    "pedestrian":    (0,   255, 0),    # green
    "rider":         (0,   200, 100),
    "car":           (255, 0,   0),    # blue
    "truck":         (255, 100, 0),
    "bus":           (255, 0,   100),
    "motorcycle":    (0,   0,   255),  # red
    "bicycle":       (0,   150, 255),
    "traffic light": (0,   255, 255),  # yellow
    "traffic sign":  (200, 0,   255),
}
DEFAULT_COLOUR = (180, 180, 180)


def draw(frame: "cv2.Mat", results, model) -> "cv2.Mat":
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf  = round(float(box.conf[0]), 2)
        colour = CLASS_COLOURS.get(label, DEFAULT_COLOUR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{label} {conf}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path,
                        default=Path("data/bdd100k/images/100k/val"),
                        help="Folder of BDD100K images to sample from")
    parser.add_argument("--output", type=Path, default=Path("output/baseline"),
                        help="Folder to save annotated images")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of images to sample")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold (lower = more detections)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    all_images = sorted(args.images.glob("*.jpg"))
    if not all_images:
        print(f"No .jpg files found in {args.images}")
        return

    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n, len(all_images)))

    print(f"Loading yolov8s.pt (COCO pretrained) ...")
    model = YOLO("yolov8s.pt")
    # GPU if available, otherwise CPU
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

    for i, img_path in enumerate(sample, 1):
        frame = cv2.imread(str(img_path))
        results = model.predict(frame, conf=args.conf, verbose=False, device=device)[0]
        annotated = draw(frame, results, model)
        out_path = args.output / img_path.name
        cv2.imwrite(str(out_path), annotated)
        n_det = len(results.boxes)
        print(f"  [{i}/{len(sample)}] {img_path.name}  →  {n_det} detections  →  {out_path.name}")

    print(f"\nDone. Open {args.output}/ to review the images.")


if __name__ == "__main__":
    main()
