import numpy as np
import cv2
import pillow_heif
import sys
import os
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import scipy.interpolate as interp

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
    
    # Ensure output dir exists
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/gray.png', gray)
    
    ret,thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    cv2.imwrite('output/thresh.png', thresh)
    return thresh

def detect_contours(image):
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
    
    while length != 4:
        if length < 4:
            var += 0.005
        else:
            var -= 0.005
        # Missing solution when the captured image is not a rectangle or shadow is clipped in
        epsilon = var * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        length = len(approx)
        
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


# --- MESH WARP UTILITY FUNCTIONS ---

def get_contour_segment(contour, idx1, idx2):
    """Extracts the shortest path along the contour loop between two indices."""
    N = len(contour)
    dist_forward = (idx2 - idx1) % N
    dist_backward = (idx1 - idx2) % N
    
    if dist_forward < dist_backward:
        if idx1 < idx2: return contour[idx1:idx2+1]
        else: return np.vstack((contour[idx1:], contour[:idx2+1]))
    else:
        if idx2 < idx1: return contour[idx2:idx1+1][::-1]
        else: return np.vstack((contour[idx2:], contour[:idx1+1]))[::-1]

def resample_curve(points, num_points):
    """Fits a B-spline to the physical curve and resamples it to uniform spacing."""
    # splprep crashes if there are exact duplicate coordinates consecutively
    diffs = np.any(points[1:] != points[:-1], axis=1)
    points = np.vstack((points[0], points[1:][diffs]))
    
    x = points[:, 0]
    y = points[:, 1]
    
    k = min(3, len(points) - 1)
    if k < 1: 
        return np.linspace(points[0], points[-1], num_points)
        
    tck, u = interp.splprep([x, y], s=0, k=k)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = interp.splev(u_new, tck)
    
    return np.column_stack((x_new, y_new))


# --- NEW MESH WARP FUNCTION ---

def mesh_warp(image, corners, contour):
    TL, TR, BR, BL = corners

    # 1. Output dimensions (Enforce A4 Aspect Ratio)
    width = max(np.linalg.norm(BR - BL), np.linalg.norm(TR - TL))
    height = width * 1.414
    width, height = int(width), int(height)
    
    # 2. Find closest indices of the 4 corners in the raw contour loop
    contour_flat = contour.reshape(-1, 2)
    corner_indices = []
    for corner in corners:
        distances = np.linalg.norm(contour_flat - corner, axis=1)
        corner_indices.append(np.argmin(distances))
        
    TL_idx, TR_idx, BR_idx, BL_idx = corner_indices
    
    # 3. Extract the physical curved edges using shortest loop paths
    top_edge_raw = get_contour_segment(contour_flat, TL_idx, TR_idx)
    # BL to BR ensures the array moves left-to-right to match the top edge
    bottom_edge_raw = get_contour_segment(contour_flat, BL_idx, BR_idx) 
    
    # 4. Resample the curves using B-Splines to perfectly match the target width
    top_edge_full = resample_curve(top_edge_raw, width)
    bottom_edge_full = resample_curve(bottom_edge_raw, width)
    
    # 5. Build the dense mapping grid (Source Mesh)
    dest_y, dest_x = np.mgrid[0:height, 0:width]
    
    # v represents vertical progression from top (0.0) to bottom (1.0)
    v = dest_y.astype(np.float32) / (height - 1)
    V = v[..., np.newaxis] # (height, width, 1)
    
    # Expand edges to broadcast vertically: shape (1, width, 2)
    TE = top_edge_full.reshape(1, width, 2)
    BE = bottom_edge_full.reshape(1, width, 2)
    
    # Linearly interpolate vertically between the curved top and bottom edges
    source_mesh = TE + V * (BE - TE)
    
    map_x = source_mesh[..., 0].astype(np.float32)
    map_y = source_mesh[..., 1].astype(np.float32)
    
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imwrite("./output/warped.png", warped_image)
    print("Mesh warping complete! Saved to ./output/warped.png")
    return warped_image
    
    
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
        return cv2.imread(str(path))

def main():
    """File Selector"""
    file_path = get_image_path()
    if not file_path:
        print("No file selected. Exiting")
        return

    """ Parse image content """
    # adding file check later for other extensions
    IMAGE = load_image(file_path)

    """Pre-process and get contours"""
    processed_img = preprocess(IMAGE)
    contours = detect_contours(processed_img)

    approx, corners, contour = detect_doc_contour(contours)
    
    draw_contours(IMAGE.copy(), [approx], "./output/hull.png")
    draw_contours(IMAGE.copy(), [contour], "./output/contour.png")
    draw_corners(IMAGE.copy(), corners)
    
    print("before warp")
    """Mesh warp and return"""
    mesh_warp(IMAGE.copy(), corners, contour)

if __name__ == '__main__':
    main()