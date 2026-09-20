"""Label ground-truth document corners for the detection benchmark.

Every image in docs/test/ without an annotation opens in the corner-adjust
window, pre-seeded with the detector's guess. Enter saves the annotation,
Esc skips the image (it stays unlabelled), Ctrl+C quits; progress is saved
after every image. Annotations live in docs/test/annotations.json.

Because the window starts from the detector's guess, move EVERY corner onto
the true page edge -- accepting a guess that is merely close inflates scores.

Usage: python -m eval.annotate [--images DIR] [--annotations FILE]
"""

import argparse
import json
from pathlib import Path

from src.file_loader import load_image
from src.contour import detect_document
from src.interactive import adjust_corners_interactive

TEST_DIR = Path(__file__).resolve().parent.parent / "docs" / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def main():
    parser = argparse.ArgumentParser(description="Annotate ground-truth document corners.")
    parser.add_argument("--images", type=Path, default=TEST_DIR)
    parser.add_argument("--annotations", type=Path, default=TEST_DIR / "annotations.json")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text()) if args.annotations.exists() else {}
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    todo = [p for p in paths if p.name not in annotations]
    print(f"{len(annotations)} annotated, {len(todo)} to go.")

    for i, path in enumerate(todo, 1):
        image = load_image(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        print(f"[{i}/{len(todo)}] {path.name}")
        corners = adjust_corners_interactive(image, detect_document(image).corners)
        if corners is None:
            continue

        annotations[path.name] = corners.tolist()
        args.annotations.write_text(json.dumps(annotations, indent=2))

    print(f"{len(annotations)} annotations in {args.annotations}")


if __name__ == "__main__":
    main()
