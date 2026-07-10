import numpy as np
import cv2

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

def detect_contours2(img):
    con = np.zeros_like(img)
   
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
    return con

def draw_contours(image, contours, output):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imwrite(output, image)

def detect_doc_contour(image):
    contours = detect_contours(image)

    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )
    
    hull = cv2.convexHull(sorted_contours[0])

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
