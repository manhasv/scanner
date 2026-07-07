import numpy as np
import cv2
import pillow_heif

INPUT = "./input/test_img2.HEIC"
RATIO = 0
IMAGE = None
def main():
    # adding file check later for other extensions
    heif_file = pillow_heif.open_heif(INPUT, convert_hdr_to_8bit=False, bgr_mode=True)
    image = np.asarray(heif_file)

    processed_img = preprocess(image)
    
    contours = detect_contours(processed_img)

    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )
    output = image.copy()
    hull = cv2.convexHull(sorted_contours[0])
    draw_contours(output, [hull], "./output/contour.png")
    print(hull.shape)

def resize_img(image):
    height, width = image.shape[:2]
    target_height = 500
    proportion = target_height/ float(height)
    target_width = int(width * proportion)
    resized = cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_AREA)
    return resized

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)

    ret,thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    output = image.copy()
    cv2.imwrite('output/thresh.png', thresh)
    # Morph Close
    # kernel = np.ones((5, 5), np.uint8) 
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    # morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("./output/morphed.jpg", morphed)
    return thresh

def detect_contours(image):
    #cv2.RETR_EXTERNAL: Retrieves only the outermost boundary contours.
    #cv2.RETR_LIST: Retrieves all contours without establishing any parent-child hierarchy.
    #cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchical relationship tree

    #cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments (e.g., reduces a rectangle boundary to just 4 corner points)
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def draw_contours(image, contours, output):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imwrite(output, image)

if __name__ == '__main__':
    main()