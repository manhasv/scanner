import cv2
import numpy as np
from src.file_loader import *
from src.preprocess import *
def main():

    file_path = get_image_path()
    img = load_image(file_path)
    print(img.shape)

    processed_img = preprocess_thresh(img.copy())
    cv2.imwrite("./output/process.png", processed_img)

    processed_img = preprocess_clahe(img.copy())
    cv2.imwrite("./output/process2.png", processed_img)

    #contours = detect_contours(processed_img)

    
if __name__ == "__main__":
    main()