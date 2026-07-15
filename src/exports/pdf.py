from io import BytesIO

import cv2
from PIL import Image


def export(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(rgb)

    buffer = BytesIO()

    pil.save(buffer, format="PDF", resolution=300)

    return (
        buffer.getvalue(),
        "application/pdf",
        ".pdf",
    )