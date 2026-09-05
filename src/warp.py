import numpy as np
import cv2

from src.file_loader import *
from src.preprocess import *
from src.contour import detect_doc_contour
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

def warp2(image, corners):

    (tl, tr, br, bl) = corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
	
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)

    return final
  
# def process(img):
#     print(img.shape)
#     processed_img = preprocess_thresh(img.copy())

#     approx, corners, contour = detect_doc_contour(processed_img)
    
#     warped = warp(img.copy(), corners)
    
#     wb = gray_world_white_balance(warped)
#     illu = illumination_correction(wb)

#     return illu

def warp_process(img, corners):
    warped = warp(img.copy(), corners)
    
    wb = gray_world_white_balance(warped)
    illu = illumination_correction(wb)

    return illu