import pillow_heif
import sys
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
import numpy as np
import cv2

__all__ = ['get_image_path', 'load_image']

def select_image_file():
    """File Picker using tkinter"""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Document Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.HEIC *.heic"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path if file_path else None

def get_image_path():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Image to scan"
    )

    args = parser.parse_args()

    if args.image:
        return args.image

    return select_image_file()

def load_image(path):
    """Loads image, handling files extension"""
    _, ext = os.path.splitext(path)
    if ext.lower() in ['.heic']:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, bgr_mode=True)
        return np.asarray(heif_file)
    else:
        return cv2.imread(path)
