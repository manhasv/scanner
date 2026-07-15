from . import raster
from . import pdf


def export_image(image, fmt):

    fmt = fmt.lower()

    if fmt == "pdf":
        return pdf.export(image)

    return raster.export(image, fmt)