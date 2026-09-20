"""Benchmark the document detector against hand-annotated ground truth.

Runs `detect_document` on each annotated image and scores the predicted
quadrilateral by its IoU with the annotated one. An image is a hit when
IoU >= --threshold; a fallback result (no candidate found) is always a miss,
since it is a guess rather than a detection.

Usage: python -m eval.evaluate [--threshold 0.9] [--debug DIR] [--images DIR] [--annotations FILE]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from src.file_loader import load_image
from src.contour import detect_document

TEST_DIR = Path(__file__).resolve().parent.parent / "docs" / "test"


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(intersection / max(union, 1.0))


def save_overlay(image: np.ndarray, truth: np.ndarray, predicted: np.ndarray,
                 label: str, path: Path, height: int = 900) -> None:
    """Draw ground truth (green) and prediction (red) on a downscaled copy."""
    scale = height / image.shape[0]
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for quad, color in ((truth, (0, 255, 0)), (predicted, (0, 0, 255))):
        pts = (np.asarray(quad, dtype=np.float32) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
    cv2.putText(canvas, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(canvas, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Score document detection against ground truth.")
    parser.add_argument("--threshold", type=float, default=0.9, help="IoU needed to count as a hit")
    parser.add_argument("--debug", type=Path, metavar="DIR",
                        help="Save overlays (green=ground truth, red=prediction) into DIR")
    parser.add_argument("--images", type=Path, default=TEST_DIR)
    parser.add_argument("--annotations", type=Path, default=TEST_DIR / "annotations.json")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    rows = []

    for name, truth in annotations.items():
        image = load_image(str(args.images / name))
        if image is None:
            print(f"Skipping unreadable image: {name}")
            continue

        start = time.perf_counter()
        detection = detect_document(image)
        elapsed = time.perf_counter() - start

        iou = quad_iou(detection.corners, np.array(truth))
        hit = iou >= args.threshold and detection.method != "fallback"
        rows.append((name, iou, detection.method, elapsed, hit))

        if args.debug:
            args.debug.mkdir(parents=True, exist_ok=True)
            label = f"{name} IoU {iou:.2f} {detection.method} conf {detection.confidence:.2f}"
            save_overlay(image, np.array(truth), detection.corners, label,
                         args.debug / f"{Path(name).stem}.jpg")

    if not rows:
        print("No annotated images to evaluate.")
        return

    print(f"{'image':<24} {'IoU':>6}  {'method':<14} {'time':>7}  result")
    for name, iou, method, elapsed, hit in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<24} {iou:6.3f}  {method:<14} {elapsed * 1000:5.0f}ms  {'hit' if hit else 'MISS'}")

    hits = sum(r[4] for r in rows)
    ious = [r[1] for r in rows]
    print()
    print(f"Detection rate @ IoU >= {args.threshold}: {hits}/{len(rows)} ({100 * hits / len(rows):.1f}%)")
    print(f"Mean IoU {statistics.mean(ious):.3f} | median IoU {statistics.median(ious):.3f} | "
          f"fallbacks {sum(r[2] == 'fallback' for r in rows)} | "
          f"median time {statistics.median(r[3] for r in rows) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
