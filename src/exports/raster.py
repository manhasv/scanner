import cv2

RASTER_FORMATS = {
    "jpg": ("image/jpeg", ".jpg"),
    "jpeg": ("image/jpeg", ".jpg"),
    "png": ("image/png", ".png"),
    "bmp": ("image/bmp", ".bmp"),
    "tiff": ("image/tiff", ".tiff"),
    "webp": ("image/webp", ".webp"),
}


def export(image, fmt):
    fmt = fmt.lower()

    if fmt not in RASTER_FORMATS:
        raise ValueError(f"Unsupported raster format: {fmt}")

    media_type, extension = RASTER_FORMATS[fmt]

    ok, encoded = cv2.imencode(extension, image)

    if not ok:
        raise RuntimeError("Failed to encode image")

    return encoded.tobytes(), media_type, extension