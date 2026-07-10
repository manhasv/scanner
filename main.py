import cv2
import numpy as np
from src.file_loader import *
from src.preprocess import *
from src.contour import detect_doc_contour, draw_contours, draw_corners
from src.warp import warp, warp2


def gray_world_white_balance(img):
    # 1. Convert to float to prevent uint8 overflow issues during math operations
    img_float = img.astype(np.float32)
    
    # 2. Calculate the average value for each color channel (Blue, Green, Red)
    # axis=(0,1) averages across rows and columns, leaving a 1D array of 3 channels
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    
    # 3. Calculate the overall baseline gray target (average of all 3 channel averages)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # 4. Compute scaling factors for each individual channel
    # Safeguard against zero-division in completely black channels
    scale_b = avg_gray / (avg_b if avg_b != 0 else 1.0)
    scale_g = avg_gray / (avg_g if avg_g != 0 else 1.0)
    scale_r = avg_gray / (avg_r if avg_r != 0 else 1.0)
    
    # 5. Apply scaling factors to equalize channel weights
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    
    # 6. Clip values to [0, 255] range and cast back to unsigned 8-bit integer
    corrected_img = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return corrected_img

def contrast_enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced_img

def main():

    file_path = get_image_path()
    img = load_image(file_path)
    print(img.shape)

    processed_img = preprocess_thresh(img.copy())
    #cv2.imwrite("./output/process.png", processed_img)

    processed_img2 = preprocess_clahe(img.copy())
    #cv2.imwrite("./output/process2.png", processed_img2)

    approx, corners, contour = detect_doc_contour(processed_img)
    print(contour.shape)
    
    # draw_contours(img.copy(), [approx], "./output/hull.png")
    # draw_contours(img.copy(), contour, "./output/contour.png")
    
    warped = warp(img.copy(), corners)
    
    wb = gray_world_white_balance(warped)
    cv2.imwrite("./output/wb.png", wb)
    contrast = contrast_enhance(wb)
    cv2.imwrite("./output/contrast.png", contrast)

    
if __name__ == "__main__":
    main()