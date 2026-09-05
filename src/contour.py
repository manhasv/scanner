from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from src.preprocess import DetectionArtifacts, prepare_detection_artifacts

@dataclass
class DocumentDetection:
    corners: np.ndarray     # float32, original-resolution TL/TR/BR/BL.
    confidence: float       # 0.0 to 1.0
    method: str             # "contour" or "min_area_rect"


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return four points in TL, TR, BR, BL order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    # Find the geometric center of the polygon
    center = np.mean(points, axis=0)
    
    # Calculate the angle of each point relative to the center
    # arctan2 returns angles from -pi to pi
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    # Sorting by angle naturally orders the points clockwise: TL, TR, BR, BL
    # (Assuming OpenCV's top-left origin coordinate system)
    return points[np.argsort(angles)]

def _is_valid_quad(quad: np.ndarray, image_area: float) -> bool:
    """Check if the 4 pts convex and has sides > 20px"""
    quad = order_corners(quad)

    if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32)):
        return False

    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if area < image_area * 0.05:
        return False

    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return bool(np.all(sides > 20))

def _edge_support(dilated_edges: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of a narrow quadrilateral outline supported by image edges."""

    boundary = np.zeros_like(dilated_edges)

    cv2.polylines(
        boundary,
        [quad.astype(np.int32).reshape(-1, 1, 2)],
        isClosed=True,
        color=255,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    boundary_pixels = np.count_nonzero(boundary)

    if boundary_pixels == 0:
        return 0.0

    return float(np.count_nonzero(cv2.bitwise_and(boundary, dilated_edges))
                / boundary_pixels)

def _geometry_score(quad: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Geo score is a combination of how big the shape occupy the image, how similar two opposite edge are 
        and how much of a 90 angle the edges are making. However, a 25% range is included in case of perspective distortion"""
    height, width = image_shape
    image_area = height * width

    # area
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    area_score = min(1.0, area / (image_area * 0.80))

    # opposite
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    ratio1 = 1.0 if ratio1 > 0.75 else ratio1
    ratio2 = 1.0 if ratio2 > 0.75 else ratio2
    
    opposite_score = 0.5 * (ratio1 + ratio2)

    # Right angle, edges should be relatively right angle for a document
    vectors = np.roll(quad, -1, axis=0) - quad
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosines = np.abs(np.sum(vectors * np.roll(vectors, -1, axis=0), axis=1))
    mean_cosine = float(np.mean(cosines))
    
    if mean_cosine < 0.45:
        right_angle_score = 1.0
    else:
        right_angle_score = max(0.0, 1.0 - ((mean_cosine - 0.45) * 2.0))

    # Frame boundary penalty. Which means a page clipped on all four sides is usually the image frame, not a page.
    border = max(5, int(min(height, width) * 0.01))
    touches = (
        (quad[:, 0] < border)
        | (quad[:, 0] > width - 1 - border)
        | (quad[:, 1] < border)
        | (quad[:, 1] > height - 1 - border)
    )
    frame_score = 0.55 if np.count_nonzero(touches) == 4 else 1.0

    return float(
        (0.45 * area_score + 0.30 * opposite_score + 0.25 *
        right_angle_score)
        * frame_score
    )


def _score_candidate(
    quad: np.ndarray,
    artifacts: DetectionArtifacts,
    dilated_edges: np.ndarray
) -> float:
    geometry = _geometry_score(quad, artifacts.edges.shape)
    support = _edge_support(dilated_edges, quad)

    # Boundary evidence is more important than rectangularity: photos may
    # contain rectangular objects, but the document boundary should be edged.

    return float(np.sqrt(geometry * support))

def _candidates_from_mask(
    mask: np.ndarray,
    image_area: float,
) -> list[tuple[np.ndarray, str]]:
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[tuple[np.ndarray, str]] = []

    for contour in contours:
        # Inspect each contour, reject the small one with < 5% area, or no valid 4 corners after hull 

        if cv2.contourArea(contour) < image_area * 0.05:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        for epsilon_ratio in (0.008, 0.012, 0.018, 0.025, 0.035):
            approximation = cv2.approxPolyDP(
                hull,
                epsilon_ratio * perimeter,
                closed=True,
            )

            if len(approximation) != 4:
                continue
            quad = order_corners(approximation)
            if _is_valid_quad(quad, image_area):
                candidates.append((quad, "contour"))

        # This recovers pages with one broken boundary while rejecting shapes
        # whose contour occupies too little of their bounding rectangle.

        rectangle = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rectangle)
        box_area = cv2.contourArea(box.reshape(-1, 1, 2))
        fill_ratio = cv2.contourArea(hull) / max(box_area, 1.0)

        if fill_ratio >= 0.65:
            quad = order_corners(box)
            if _is_valid_quad(quad, image_area):
                candidates.append((quad, "min_area_rect"))

    return candidates

def _deduplicate(
    candidates: list[tuple[np.ndarray, str]],
) -> list[tuple[np.ndarray, str]]:
    """Compare candidates against others using Intersection over Union IoU
    , if overlap over 92%, count as dup and discard candidate"""
    unique: list[tuple[np.ndarray, str]] = []

    for quad, method in candidates:
        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
        duplicate = False
        for known, _ in unique:
            known_area = cv2.contourArea(known.reshape(-1, 1, 2))
            intersection, _ = cv2.intersectConvexConvex(quad, known)
            iou = intersection / max(quad_area + known_area - intersection, 1.0)

            if iou > 0.92:
                duplicate = True
                break

        if not duplicate:
            unique.append((quad, method))

    return unique


def _fallback_corners(width: int, height: int) -> np.ndarray:
    """Safe, editable proposal when no credible document was found."""
    print("No candidate found, resort to Fallback")
    margin = max(10, int(min(width, height) * 0.04))
    return np.array(
        [
            [margin, margin],
            [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin],
            [margin, height - 1 - margin],
        ],
        dtype=np.float32,
    )

def detect_document(image_bgr: np.ndarray, debug: bool = False) -> DocumentDetection:
    """
    Detect the most likely document quadrilateral.
    Fold lines appear as interior edges and do not become candidates unless
    they also form a large, well-supported outer quadrilateral.
    """

    artifacts = prepare_detection_artifacts(image_bgr)

    # this will crash in uvicorn since it's not on primary thread. 
    # if debug:
    #     debug_views = {
    #         #"1 - Working Image": artifacts.image,
    #         "2 - Lightness": artifacts.lightness,
    #         "3 - Normalized": artifacts.normalized,
    #         "4 - Edges": artifacts.edges,
    #         "5 - Paper Mask": artifacts.masks[1]
    #     }
    #     display_height = 600
        
    #     for name, img in debug_views.items():
    #         # Calculate aspect ratio to maintain image proportions
    #         h, w = img.shape[:2]
    #         aspect_ratio = w / h
    #         display_width = int(display_height * aspect_ratio)
    #         # Adjust display window
    #         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #         cv2.resizeWindow(name, display_width, display_height)
    #         cv2.imshow(name, img)
    #     cv2.waitKey(0)

    height, width = artifacts.edges.shape
    image_area = float(height * width)
    candidates: list[tuple[np.ndarray, str]] = []

    for mask in artifacts.masks:
        candidates.extend(_candidates_from_mask(mask, image_area))
    candidates = _deduplicate(candidates)

    if debug:
        # Draw ALL surviving candidates in bright green
        preview = artifacts.image.copy()
        for quad, _ in candidates:
            int_quad = quad.astype(np.int32)
            cv2.polylines(preview, [int_quad], True, (0, 255, 0), 2)
        #cv2.imshow("name", preview)
        cv2.imwrite("debug_all_candidates.jpg", preview)

    if not candidates:
        return DocumentDetection(
            corners=_fallback_corners(image_bgr.shape[1],
            image_bgr.shape[0]),
            confidence=0.0,
            method="fallback",
        )

    dilated_edges = cv2.dilate(
        artifacts.edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    best_quad, best_method = max(
        candidates,
        key=lambda item: _score_candidate(item[0], artifacts, dilated_edges),
    )

    confidence = _score_candidate(best_quad, artifacts, dilated_edges)

    original_corners = order_corners(
        best_quad * artifacts.scale_to_original
    )

    return DocumentDetection(
        corners=original_corners,
        confidence=confidence,
        method=best_method,
    )

def detect_doc_contour(image_bgr: np.ndarray):
    return detect_document(image_bgr).corners