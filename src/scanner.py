from src.file_loader import *
from src.preprocess import *
from src.contour import detect_doc_contour, draw_contours, draw_corners
from src.warp import warp, warp2
from src.postprocess import *
import cv2

def process(img):
    print(img.shape)
    processed_img = preprocess_thresh(img.copy())
    #cv2.imwrite("./output/process.png", processed_img)

    processed_img2 = preprocess_clahe(img.copy())
    #cv2.imwrite("./output/process2.png", processed_img2)

    approx, corners, contour = detect_doc_contour(processed_img)
    
    warped = warp(img.copy(), corners)
    
    wb = gray_world_white_balance(warped)
    illu = illumination_correction(wb)

    cv2.imwrite("./output/output.png", illu)
    return illu