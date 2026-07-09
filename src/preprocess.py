import cv2
import numpy as np

__all__ = ['preprocess_thresh', 'preprocess_clahe', 'preprocess_cannyedge']

def remove_text(img):
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)

    return img

def preprocess_thresh(image):
    image = remove_text(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)
    _,thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    output = image.copy()

    # Morph Close
    # kernel = np.ones((5, 5), np.uint8) 
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    # morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return thresh

def preprocess_clahe(image):
    image = remove_text(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast (handles shadows much better)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)
    cv2.imwrite("output/01_clahe.png", gray)

    # Remove small noise while preserving edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150
    )

    # Connect broken edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return closed

def preprocess_cannyedge(image):
    # This one is quite bad right now
    image = remove_text(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return canny