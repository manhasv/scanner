import pillow_heif
import os
import numpy as np
import cv2

__all__ = ['load_image']

def load_image(path):
    """Loads image, handling files extension"""
    _, ext = os.path.splitext(path)
    if ext.lower() in ['.heic']:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, bgr_mode=True)
        return np.asarray(heif_file).copy()
    else:
        return cv2.imread(path)
