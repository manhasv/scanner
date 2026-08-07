from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class DetectionArtifacts:
    image: np.ndarray       # Detection-resolution BGR image.
    lightness: np.ndarray   # Lab L channel.
    normalized: np.ndarray  # Shadow-normalized L channel.
    edges: np.ndarray       # Cleaned Canny edge map.
    masks: list[np.ndarray] # Binary masks used for contour candidates.
    scale_to_original: float

def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, value)
    return value if value % 2 else value + 1

def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return clean

def _remove_fragments(img: np.ndarray) -> np.ndarray:
    # Repeated Closing operation to remove fragments like text from the document, or noise in background
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
    return img

def _auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    upper = max(upper, lower + 20)
    return cv2.Canny(image, lower, upper, L2gradient=True)

def prepare_detection_artifacts(
    image_bgr: np.ndarray,
    max_dimension: int = 1600,
) -> DetectionArtifacts:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty BGR image")

    original_height, original_width = image_bgr.shape[:2]
    longest_side = max(original_height, original_width)
    detection_scale = min(1.0, max_dimension / longest_side)

    if detection_scale < 1.0:
        working = cv2.resize(
            image_bgr,
            None,
            fx=detection_scale,
            fy=detection_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image_bgr.copy()

    height, width = working.shape[:2]

    # 1. Convert to LAB color space and extract L channel
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    # 2. Illumination Normalization via float division with Gaussian background
    light_float = lightness.astype(np.float32)
    blur_size = _odd(int(min(height, width) * 0.12), minimum=31)
    illumination = cv2.GaussianBlur(light_float, (blur_size, blur_size), 0)
    illumination = np.maximum(illumination, 10.0) #

    # Scale relative to average lightness to prevent integer saturation at 255
    mean_lightness = float(np.mean(light_float))
    normalized_float = (light_float / illumination) * mean_lightness
    normalized = np.clip(normalized_float, 0, 255).astype(np.uint8)

    # 3. Enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)

    # 4. Edge-preserving smoothing
    denoised = cv2.bilateralFilter(
        normalized, d=7, sigmaColor=45, sigmaSpace=45
    )

    # Extract the Saturation channel to find objects with color
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Threshold: Low saturation (0-45) becomes white paper (255), 
    # High saturation (colorful backgrounds) becomes black (0).
    _, sat_mask = cv2.threshold(saturation, 45, 255, cv2.THRESH_BINARY_INV)
    
    # Close small holes where colored logos or colored ink might exist on the paper
    sat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, sat_kernel)

    # 5. Canny edge extraction & morphological closing
    raw_edges = _auto_canny(denoised)
    close_size = _odd(int(min(height, width) * 0.008), minimum=3)

    # horizontal = cv2.getStructuringElement(
    #     cv2.MORPH_RECT, (close_size * 3, close_size)
    # )
    # vertical = cv2.getStructuringElement(
    #     cv2.MORPH_RECT, (close_size, close_size * 3)
    # )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    edges = cv2.morphologyEx(raw_edges, cv2.MORPH_CLOSE, kernel) # use horizontal and vertical if need to bridge specific lines
    
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    edges = cv2.bitwise_and(edges, edges, mask=sat_mask)

    # 6. Adaptive Thresholding Paper Mask
    block_size = _odd(int(min(height, width) * 0.06), minimum=31)
    paper_mask = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C=2,  # pixel is classified as white if Pixel > Mean - C
    )
    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_size * 2, close_size * 2)
        ),
    )

    paper_mask = cv2.bitwise_and(paper_mask, sat_mask)

    paper_mask = _remove_small_components(
        paper_mask,
        min_area=max(100, int(height * width * 0.01)), # close small fragments that is 1% of the image
    )

    return DetectionArtifacts(
        image=working,
        lightness=lightness,
        normalized=normalized,
        edges=edges,
        masks=[edges, paper_mask],
        scale_to_original=1.0 / detection_scale,
    )