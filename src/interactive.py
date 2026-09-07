from __future__ import annotations

import math

import cv2
import numpy as np

from src.contour import order_corners

"""OpenCV-window corner adjustment for new cli. 
Adapted from the drag-corner prototype in docs/archive/scripts/manual_corners.py."""


GRAB_RADIUS = 25


def adjust_corners_interactive(
    image: np.ndarray,
    initial_corners: np.ndarray,
    display_height: int = 900,
    window_name: str = "Adjust corners (drag corners, Enter=confirm, Esc=cancel)",
) -> np.ndarray | None:
    """Opens a window to fine-tune a detected document quadrilateral.

    `initial_corners` must be original-resolution TL/TR/BR/BL, as returned
    by `detect_document`. Returns adjusted corners in the same convention,
    or None if the user cancels with Esc.
    """
    height = image.shape[0]
    # Always scale to display_height (up or down) so the window opens at a
    # consistent, comfortable size regardless of the source photo's resolution.
    scale = display_height / height
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    corners = (np.asarray(initial_corners, dtype=np.float32) * scale).tolist()
    state = {"dragging": False, "active": -1}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cx, cy) in enumerate(corners):
                if math.hypot(x - cx, y - cy) < GRAB_RADIUS:
                    state["dragging"] = True
                    state["active"] = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            corners[state["active"]] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            state["active"] = -1

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    confirmed = False
    try:
        while True:
            canvas = display.copy()
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for i, (cx, cy) in enumerate(corners):
                color = (0, 0, 255) if state["active"] == i else (255, 0, 0)
                cv2.circle(canvas, (int(cx), int(cy)), 10, color, -1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                confirmed = True
                break
            if key == 27:  # Esc
                break
    finally:
        cv2.destroyWindow(window_name)

    if not confirmed:
        return None

    return order_corners(np.array(corners, dtype=np.float32) / scale)
