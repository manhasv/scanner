import numpy as np
import cv2

from src.file_loader import *
from src.preprocess import *
from src.postprocess import *

def warp(image, corners):
    corners = np.asarray(corners, dtype=np.float32)
    TL, TR, BR, BL = corners

    width = max(np.linalg.norm(BR - BL), np.linalg.norm(TR - TL))
    height = width * 1.414

    matrix = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    M = cv2.getPerspectiveTransform(corners, matrix, solveMethod=None)
    
    warped_image = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped_image

def warp_process(img, corners):
    warped = warp(img.copy(), corners)
    
    wb = gray_world_white_balance(warped)
    illu = illumination_correction(wb)

    return illu