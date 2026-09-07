"""Command-line entrypoint for the document scanner: detect the document
boundary, let the user adjust corners, warp, and export -- all against a
local file."""

import argparse
from pathlib import Path

from src.file_loader import load_image
from src.tui_picker import select_image_file_tui
from src.contour import detect_document
from src.interactive import adjust_corners_interactive
from src.warp import warp_process
from src.exports import export_image


def main():
    parser = argparse.ArgumentParser(description="Scan a document image from the command line.")
    parser.add_argument(
        "image", nargs="?", type=Path,
        help="Image to scan (opens a terminal file browser if omitted)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file path (default: <input>_scan.jpg)",
    )
    parser.add_argument(
        "-f", "--format",
        help="Output format: jpg/png/bmp/tiff/webp/pdf (default: inferred from --output, else jpg)",
    )
    args = parser.parse_args()

    image_path = args.image or select_image_file_tui()

    if not image_path:
        print("No image selected.")
        return

    image_path = Path(image_path)
    image = load_image(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    doc = detect_document(image)
    print(f"Detected document (confidence {doc.confidence:.2f}, method {doc.method}).")

    print("Opening corner-adjustment window (drag corners, Enter to confirm, Esc to cancel)...")
    corners = adjust_corners_interactive(image, doc.corners)
    if corners is None:
        print("Cancelled.")
        return

    result = warp_process(image, corners)

    output_path = args.output or image_path.with_name(f"{image_path.stem}_scan.jpg")
    fmt = args.format or output_path.suffix.lstrip(".") or "jpg"

    data, _, extension = export_image(result, fmt)
    if output_path.suffix.lower() != extension:
        output_path = output_path.with_suffix(extension)

    output_path.write_bytes(data)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
