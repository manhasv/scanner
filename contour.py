import numpy as np
import cv2
import pillow_heif
import sys
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def resize_img(image):
    height, width = image.shape[:2]
    target_height = 500
    proportion = target_height/ float(height)
    target_width = int(width * proportion)
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)
    cv2.imwrite('output/gray.png', gray)
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

def detect_doc_contour(contours):
    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )
    
    hull = cv2.convexHull(sorted_contours[0])
    print(f"contour {sorted_contours[0].shape}")
    print(f"hull {hull.shape}")

    # approx to curve it down to 4 corners
    length = 0
    perimeter = cv2.arcLength(hull, True)
    var = 0.02
    
    # while length != 4:
    #     if length < 4:
    #         var += 0.005
    #     else:
    #         var -= 0.005
    # Missing solution when the captured image is not a rectangle or shadow is clipped in
    epsilon = var * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) != 4:
        print(f"Error length of detected contour is {len(approx)}")
        sys.exit(-1)
    
    corners = detect_corners(approx)

    return approx, corners, sorted_contours[0]
    
def detect_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")

    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-Right will have the smallest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom-Left will have the largest difference
    rect[3] = pts[np.argmax(diff)]

    return rect
    
def draw_corners(output, corners):
    for i, corner in enumerate(corners):
        x = int(corner[0])
        y = int(corner[1])
        cv2.circle(output, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        text_position = (x + 10, y - 10)
    
        cv2.putText(
            output, 
            text=str(i + 1),
            org=text_position,  
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.6,     
            color=(0, 0, 255),  
            thickness=2  
        )
    
    cv2.imwrite("output/corners.png", output)

def warp(image, corners):
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
    cv2.imwrite("./output/warped.png", warped_image)
    
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

def main():
    """File Selector"""
    file_path = get_image_path()
    if not file_path:
        print("No file selected. Exiting")
        return

    """ Parse image contnet """
    # adding file check later for other extensions
    heif_file = pillow_heif.open_heif(file_path, convert_hdr_to_8bit=False, bgr_mode=True)
    image = np.asarray(heif_file)
    IMAGE = image.copy()

    """Pre-process and get contours"""
    processed_img = preprocess(IMAGE)
    contours = detect_contours(processed_img)

    approx, corners, contour = detect_doc_contour(contours)
    
    draw_contours(IMAGE.copy(), [approx], "./output/hull.png")
    draw_contours(IMAGE.copy(), contour, "./output/contour.png")
    draw_corners(IMAGE.copy(), corners)
    
    """Perspective warp and return"""
    warp(IMAGE.copy(), corners)

if __name__ == '__main__':
    main()