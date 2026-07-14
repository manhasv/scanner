import numpy as np
import cv2
import pillow_heif
import tkinter as tk
from tkinter import filedialog
import math
import os

# ==========================================
# PHASE 3: Global State Management
# ==========================================
# We store these at the top level so our mouse callback can access and modify them
dragging = False
active_corner = -1
corners = []

def select_image_file():
    """PHASE 1: The File Picker using tkinter"""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Document Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.HEIC *.heic"), ("All files", "*.*")]
    )
    return file_path

def load_image(path):
    """Loads image, handling HEIC files natively based on your previous script."""
    _, ext = os.path.splitext(path)
    if ext.lower() in ['.heic']:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, bgr_mode=True)
        return np.asarray(heif_file).copy()
    else:
        return cv2.imread(path)

def resize_for_ui(image, max_height=800):
    """Resizes the image so it actually fits on a laptop screen for editing."""
    height, width = image.shape[:2]
    if height > max_height:
        proportion = max_height / float(height)
        target_width = int(width * proportion)
        return cv2.resize(image, (target_width, max_height), interpolation=cv2.INTER_AREA), proportion
    return image, 1.0

def mouse_callback(event, x, y, flags, param):
    """PHASE 2 & 3: OpenCV's Mouse Callback System & Dragging Logic"""
    global corners, dragging, active_corner
    
    # 1. User clicks down
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check Euclidean distance to find if we clicked near a corner
        for i, corner in enumerate(corners):
            cx, cy = corner
            distance = math.hypot(x - cx, y - cy)
            if distance < 25:  # 25 pixel grab radius
                dragging = True
                active_corner = i
                break
                
    # 2. User drags the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and active_corner != -1:
            # Update the coordinates of the grabbed corner
            corners[active_corner] = [x, y]
            
    # 3. User releases the click
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        active_corner = -1

def order_points(pts):
    """Sorts corners into Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def warp(image, ordered_corners):
    """Your existing warp function!"""
    TL, TR, BR, BL = ordered_corners

    # Calculate target dimensions using Euclidean distance
    width_top = np.linalg.norm(TR - TL)
    width_bottom = np.linalg.norm(BR - BL)
    width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(BL - TL)
    height_right = np.linalg.norm(BR - TR)
    height = max(int(height_left), int(height_right))

    matrix = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    
    M = cv2.getPerspectiveTransform(ordered_corners, matrix)
    warped_image = cv2.warpPerspective(image, M, (width, height))
    return warped_image

def main():
    global corners

    # 1. Get the file from the user
    file_path = select_image_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    # 2. Load and resize for the UI
    original_img = load_image(file_path)
    if original_img is None:
        print("Error loading image.")
        return
        
    ui_img, ratio = resize_for_ui(original_img)
    ui_height, ui_width = ui_img.shape[:2]

    # 3. Set default corners (10% margin from the edges)
    margin_x = int(ui_width * 0.1)
    margin_y = int(ui_height * 0.1)
    corners = [
        [margin_x, margin_y],                         # Top-Left
        [ui_width - margin_x, margin_y],              # Top-Right
        [ui_width - margin_x, ui_height - margin_y],  # Bottom-Right
        [margin_x, ui_height - margin_y]              # Bottom-Left
    ]

    # 4. Setup OpenCV Window and Callback
    window_name = "Interactive Scanner - Drag Corners & Press ENTER"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # ==========================================
    # PHASE 4: The Render Loop
    # ==========================================
    while True:
        # Create a fresh canvas every frame so previous drawings don't smear
        canvas = ui_img.copy()

        # Draw the polygon connecting the 4 corners
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw interactive circles at each corner
        for i, corner in enumerate(corners):
            # Highlight the corner being dragged in red
            color = (0, 0, 255) if (dragging and i == active_corner) else (255, 0, 0)
            cv2.circle(canvas, tuple(corner), radius=10, color=color, thickness=-1)

        cv2.imshow(window_name, canvas)

        # Listen for keyboard presses
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # 13 is the 'Enter' key
            print("Processing warp...")
            break
        elif key == 27: # 27 is the 'ESC' key
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # 5. Math! Scale coordinates back to original High-Res image size
    ordered_ui_corners = order_points(corners)
    high_res_corners = ordered_ui_corners / ratio

    # 6. Perform the final warp on the original image
    final_scanned_image = warp(original_img, high_res_corners)
    
    # Save and show result
    cv2.imwrite("./output/interactive_warped.png", final_scanned_image)
    print("Saved to ./output/interactive_warped.png")
    
    # Show a small preview of the result
    preview, _ = resize_for_ui(final_scanned_image)
    cv2.imshow("Final Scan", preview)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()