"""
ROI (Region of Interest) Detection for Footprint Biometric Recognition.

The foot is segmented from the background and the biometrically relevant
plantar region (ball of foot + medial arch) is extracted for downstream
feature extraction and matching.

Real acquisition data is heterogeneous: clean ink impressions on paper,
flatbed scanner captures, close-up camera photographs and phone photographs.
A single Otsu threshold is not sufficient for all of them, so this module
builds several candidate masks (Otsu, fixed threshold, adaptive threshold,
skin colour and — as a last resort — GrabCut), scores each mask for
"foot-likeness", and keeps the most plausible one.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
import numpy as np


# GrabCut initialises its Gaussian mixture models with ``cv2.kmeans``, which
# draws from OpenCV's global RNG.  Without a fixed seed the same photo produced
# a different mask (and therefore a different ROI, enhancement and prediction)
# on every call: repeated identifications of one image could disagree, and the
# stored enrollment crops could never be reproduced.  Seed it deterministically
# and serialise the call so concurrent requests cannot interleave the RNG.
_GRABCUT_LOCK = threading.Lock()
_GRABCUT_SEED = 0


# ---------------------------------------------------------------------------
# Mask generation and scoring
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove salt noise and close small gaps in a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def _skin_masks(image: np.ndarray) -> list[np.ndarray]:
    """Return conservative skin-colour masks in YCrCb and HSV spaces."""
    masks: list[np.ndarray] = []

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    masks.append(
        cv2.inRange(ycrcb, np.array([20, 130, 72]), np.array([255, 185, 140]))
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([25, 255, 255]))
    upper = cv2.inRange(hsv, np.array([155, 20, 20]), np.array([180, 255, 255]))
    masks.append(cv2.bitwise_or(lower, upper))

    return masks


def _candidate_masks(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build segmentation candidates for light-background scans and photos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates: list[tuple[str, np.ndarray]] = []

    # The classic dark-footprint-on-light-paper case from the specification.
    _, otsu_inverse = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    candidates.append(("otsu_inverse", otsu_inverse))

    # Bright objects on a dark background (some close-up camera images).
    _, otsu_normal = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    candidates.append(("otsu_normal", otsu_normal))

    # Scanner captures often have very bright backgrounds; a fixed threshold
    # catches more of the faint heel/toe edges than Otsu alone.
    if float(np.median(gray)) > 180.0:
        _, fixed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        candidates.append(("fixed_200", fixed))

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )
    candidates.append(("adaptive", adaptive))

    for index, skin in enumerate(_skin_masks(image)):
        candidates.append((f"skin_{index}", skin))

    return candidates


def _local_texture(gray: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Local standard deviation, used to tell creased skin from smooth walls."""
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (kernel_size, kernel_size))
    mean_square = cv2.blur(values * values, (kernel_size, kernel_size))
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def _score_contour(
    name: str,
    mask: np.ndarray,
    contour: np.ndarray,
    gray: np.ndarray,
    texture: np.ndarray | None = None,
    skin: np.ndarray | None = None,
) -> float:
    """Heuristic score describing how much a contour looks like a footprint."""
    height, width = mask.shape[:2]
    area = float(cv2.contourArea(contour))
    if area < 100:
        return -1e9

    x, y, box_width, box_height = cv2.boundingRect(contour)
    image_area = float(height * width)
    area_ratio = area / image_area
    bbox_ratio = float(box_width * box_height) / image_area
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    aspect = max(box_width, box_height) / max(1.0, float(min(box_width, box_height)))

    # Fraction of contour points that sit on the image border.
    points = contour.reshape(-1, 2)
    border_hits = np.sum(
        (points[:, 0] <= 2)
        | (points[:, 0] >= width - 3)
        | (points[:, 1] <= 2)
        | (points[:, 1] >= height - 3)
    )
    border_contact = float(border_hits) / max(1, len(points))

    mask_bool = mask > 0
    inside = gray[mask_bool]
    outside = gray[~mask_bool]
    contrast = abs(float(inside.mean()) - float(outside.mean())) if inside.size and outside.size else 0.0

    # Crease texture: a foot has creases, while a wall/door/floor patch is
    # smooth.  This is computed inside the contour, not the whole candidate.
    if texture is None:
        texture = _local_texture(gray, 15)
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)
    inside_texture = 0.0
    if cv2.countNonZero(contour_mask):
        inside_texture = float(texture[contour_mask > 0].mean())
    score_texture = 2.0 * min(inside_texture / 20.0, 1.0)
    if inside_texture < 8.0:
        score_texture -= 0.8
    # Skin colour: separates a bare foot from a textured tiled floor.  Combined
    # with the texture term this gives skin+creases (foot) while rejecting both
    # the smooth skin-coloured wall and the untextured non-skin floor.
    score_skin = 0.0
    if skin is not None and cv2.countNonZero(contour_mask):
        skin_fraction = float(np.count_nonzero(skin[contour_mask > 0])) / float(
            cv2.countNonZero(contour_mask)
        )
        score_skin = 1.6 * min(1.0, skin_fraction)

    # Centre preference: feet are normally near the middle of the capture.
    centre_x = x + box_width / 2.0
    centre_y = y + box_height / 2.0
    centre_distance = math.hypot(centre_x - width / 2.0, centre_y - height / 2.0)
    centre_distance /= max(1.0, 0.5 * math.hypot(width, height))

    score = 0.0
    score += 2.2 * min(area_ratio / 0.30, 1.0)
    if area_ratio < 0.02:
        score -= 2.5
    if area_ratio > 0.94:
        score -= 3.0
    if bbox_ratio > 0.97:
        score -= 2.0

    score += 1.6 * solidity
    score += 1.0 * (1.0 - min(1.0, border_contact * 2.0))
    if 1.15 <= aspect <= 4.5:
        score += 1.0
    elif aspect > 6.0:
        score -= 1.2
    score += 0.9 * min(contrast / 55.0, 1.0)
    score += score_texture
    score += score_skin
    score -= 0.6 * min(1.0, centre_distance)

    if name.startswith("skin"):
        score += 0.4
    if border_contact > 0.30:
        score -= 1.5
    return score


def _collect_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= 100]


def _grabcut_mask(image: np.ndarray, margin: float = 0.12) -> np.ndarray | None:
    """Run GrabCut initialised with a central rectangle.

    The rectangle is inset by ``margin`` on every side, which encodes the
    assumption that the foot is the dominant object near the middle of a phone
    capture.  Large images are downscaled first so this stays fast.
    """
    scale = 1.0
    working = image
    if max(image.shape[:2]) > 700:
        scale = 700.0 / max(image.shape[:2])
        working = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    height, width = working.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    margin_x = max(2, int(width * margin))
    margin_y = max(2, int(height * margin))
    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )
    try:
        with _GRABCUT_LOCK:
            cv2.setRNGSeed(_GRABCUT_SEED)
            cv2.grabCut(
                working,
                mask,
                rectangle,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_RECT,
            )
    except cv2.error:
        return None
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = _clean_mask(result, 7)
    if scale != 1.0:
        result = cv2.resize(
            result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return result


def _bbox_is_implausible(contour: np.ndarray, shape: tuple[int, int]) -> bool:
    """True when a contour spans so much of the frame that it is background.

    A wall, door or floor region often wins the area-based score because it is
    simply big.  A real foot occupies a compact, elongated part of the frame.
    """
    height, width = shape[:2]
    _, _, box_width, box_height = cv2.boundingRect(contour)
    if box_height >= 0.90 * height or box_width >= 0.90 * width:
        return True
    if box_width * box_height >= 0.82 * width * height:
        return True
    return False


def footprint_quality(
    image: np.ndarray, mask: np.ndarray, contour: np.ndarray
) -> dict:
    """Describe how much the selected region looks like a real footprint.

    The segmentation stage always returns *some* contour, even for a blank page
    or a photo of a wall.  This quality report is the guard that distinguishes
    a genuine footprint from an arbitrary blob so that non-footprint uploads can
    be rejected before they reach the classifier.
    """
    height, width = image.shape[:2]
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    area = float(cv2.contourArea(contour))
    image_area = float(height * width)
    x, y, box_width, box_height = cv2.boundingRect(contour)
    box_area = float(max(1, box_width * box_height))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    perimeter = float(cv2.arcLength(contour, True))

    mask_bool = mask > 0
    inside = gray[mask_bool]
    outside = gray[~mask_bool]
    contrast = (
        abs(float(inside.mean()) - float(outside.mean()))
        if inside.size and outside.size
        else 0.0
    )

    texture = _local_texture(gray, 15)
    inside_texture = (
        float(texture[mask_bool].mean()) if np.any(mask_bool) else 0.0
    )

    points = contour.reshape(-1, 2)
    border_hits = np.sum(
        (points[:, 0] <= 2)
        | (points[:, 0] >= width - 3)
        | (points[:, 1] <= 2)
        | (points[:, 1] >= height - 3)
    )

    quality = {
        "area": area,
        "area_ratio": area / image_area if image_area else 0.0,
        "bbox_ratio": (box_width * box_height) / image_area if image_area else 0.0,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "extent": area / box_area,
        "aspect": max(box_width, box_height) / max(1.0, float(min(box_width, box_height))),
        "circularity": (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0,
        "border_contact": float(border_hits) / max(1, len(points)),
        "contrast": contrast,
        "texture": inside_texture,
    }

    reasons: list[str] = []
    if quality["area_ratio"] < 0.02:
        reasons.append("foreground too small")
    # NOTE: a legitimate close-up photo (and an already-cropped ROI) can fill
    # most of the frame, so "covers the whole frame" is intentionally not a
    # rejection reason.  Uniform images are still rejected by the contrast,
    # outline and border-contact checks below.
    if quality["contrast"] < 3.0:
        reasons.append("no foreground/background contrast")
    if quality["aspect"] < 1.03 or quality["aspect"] > 6.0:
        reasons.append("implausible footprint aspect ratio")
    if quality["solidity"] < 0.45:
        reasons.append("implausible footprint shape")
    if quality["extent"] < 0.20 or quality["extent"] > 0.995:
        reasons.append("implausible footprint outline")
    if quality["circularity"] > 0.80:
        reasons.append("region is too round to be a footprint")
    if quality["border_contact"] > 0.5:
        reasons.append("region touches the image border")

    quality["is_footprint"] = not reasons
    quality["reasons"] = reasons
    return quality


def segment_foot(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Segment the foot region from the background.

    Args:
        image: BGR footprint image.

    Returns:
        ``(binary_mask, cleaned_threshold, contours)`` where ``binary_mask``
        is the filled mask of the selected foot contour, ``cleaned_threshold``
        is the cleaned candidate mask it came from, and ``contours`` contains
        all external contours from that mask.

    Raises:
        ValueError: If no plausible foot contour is found.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A non-empty BGR footprint image is required.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ROI detection expects a 3-channel BGR footprint image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = _local_texture(gray, 15)
    skin_mask = np.zeros(gray.shape, np.uint8)
    for candidate_skin in _skin_masks(image):
        skin_mask = cv2.bitwise_or(skin_mask, candidate_skin)
    best: tuple[float, str, np.ndarray, np.ndarray, list] | None = None

    for name, raw_mask in _candidate_masks(image):
        cleaned = _clean_mask(raw_mask, 5)
        contours = _collect_contours(cleaned)
        if not contours:
            continue
        # The foot is not always the largest blob (a wall can win on area), so
        # score every significant contour, not just the biggest one.
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            score = _score_contour(
                name, cleaned, contour, gray, texture=texture, skin=skin_mask
            )
            if best is None or score > best[0]:
                best = (score, name, cleaned, contour, contours)

    # GrabCut with a central prior is comparatively expensive, so run it when
    # the cheap masks are missing, weak, or describe a background-sized region.
    needs_grabcut = (
        best is None
        or best[0] < 2.5
        or _bbox_is_implausible(best[3], gray.shape)
    )
    if needs_grabcut:
        grabcut = _grabcut_mask(image)
        if grabcut is not None:
            contours = _collect_contours(grabcut)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                score = _score_contour(
                    "grabcut", grabcut, contour, gray, texture=texture, skin=skin_mask
                )
                use_grabcut = not _bbox_is_implausible(contour, gray.shape) and (
                    best is None
                    or _bbox_is_implausible(best[3], gray.shape)
                    or score > best[0]
                )
                if use_grabcut:
                    best = (score, "grabcut", grabcut, contour, contours)

    if best is None or best[0] < -1e8:
        raise ValueError(
            "No contours found in the image. The foot could not be segmented."
        )

    _, selected_name, cleaned, largest_contour, contours = best
    if cv2.contourArea(largest_contour) < 100:
        raise ValueError("Detected footprint region is too small to extract an ROI.")

    binary_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(binary_mask, [largest_contour], -1, 255, cv2.FILLED)

    # Keep the selected candidate available for diagnostics (tests and UI
    # expect the second return value to be a displayable threshold mask).
    return binary_mask, cleaned, list(contours)


# ---------------------------------------------------------------------------
# Anatomical partitioning and ROI extraction
# ---------------------------------------------------------------------------

def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def extract_roi(
    image: np.ndarray, mask: np.ndarray, contour: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract the midfoot/arch ROI (40%–75% of the foot bounding box).

    Returns the masked 224×224 ROI, the matching 224×224 foot-support mask and
    a dictionary describing the anatomical partitions, including the original
    bounding box of the foot.  The support mask is what lets later stages keep
    creases while hard-zeroing everything that is not foot.
    """
    x, y, width, height = cv2.boundingRect(contour)
    if width < 2 or height < 2:
        raise ValueError("Foot contour is too small to partition.")

    forefoot_y = y + int(0.40 * height)
    midfoot_y = y + int(0.75 * height)
    heel_y = y + height

    roi_y_start = _clamp(forefoot_y, 0, mask.shape[0] - 1)
    roi_y_end = _clamp(midfoot_y, roi_y_start + 1, mask.shape[0])
    roi_x_start = _clamp(x, 0, mask.shape[1] - 1)
    roi_x_end = _clamp(x + width, roi_x_start + 1, mask.shape[1])

    roi_crop = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()
    mask_crop = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    roi_masked = cv2.bitwise_and(roi_crop, roi_crop, mask=mask_crop)
    if roi_masked.size == 0:
        raise ValueError("The computed ROI is empty.")
    roi_resized = cv2.resize(roi_masked, (224, 224), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(mask_crop, (224, 224), interpolation=cv2.INTER_NEAREST)
    roi_mask = (roi_mask > 0).astype(np.uint8) * 255

    region_info = {
        "bounding_box": (x, y, width, height),
        "forefoot_y": forefoot_y,
        "midfoot_y": midfoot_y,
        "heel_y": heel_y,
        "roi_bounds": (roi_x_start, roi_y_start, roi_x_end - roi_x_start, roi_y_end - roi_y_start),
    }
    return roi_resized, roi_mask, region_info


def visualize_regions(
    image: np.ndarray, contour: np.ndarray, region_info: dict
) -> np.ndarray:
    """Draw the foot contour and the forefoot / midfoot / heel boundaries."""
    annotated = image.copy()
    x, y, width, height = region_info["bounding_box"]
    forefoot_y = region_info["forefoot_y"]
    midfoot_y = region_info["midfoot_y"]

    cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
    cv2.line(annotated, (x, forefoot_y), (x + width, forefoot_y), (255, 0, 0), 2)
    cv2.line(annotated, (x, midfoot_y), (x + width, midfoot_y), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.8, width / 500.0))
    thickness = 2
    label_x = _clamp(x + 5, 5, max(5, image.shape[1] - 220))

    cv2.putText(
        annotated,
        "Forefoot",
        (label_x, _clamp(y + int(0.20 * height), 20, image.shape[0] - 10)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    cv2.putText(
        annotated,
        "Midfoot/Arch (ROI)",
        (label_x, _clamp(y + int(0.575 * height), 20, image.shape[0] - 10)),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
    )
    cv2.putText(
        annotated,
        "Heel",
        (label_x, _clamp(y + int(0.875 * height), 20, image.shape[0] - 10)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    return annotated


def process_footprint(
    image_path_or_array, return_visualization: bool = False
) -> dict:
    """Main pipeline for footprint ROI extraction.

    Accepts a file path (``str`` or :class:`pathlib.Path`) or a BGR NumPy
    array.  Returns the 224×224 ROI, the binary foot mask, region metadata and
    (optionally) an annotated visualization.
    """
    try:
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
            if image is None:
                raise ValueError(
                    f"Could not load image from path: {image_path_or_array}"
                )
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array.copy()
        else:
            raise ValueError(
                "Input must be a file path (str/Path) or a numpy array (BGR image)."
            )

        mask, cleaned_threshold, contours = segment_foot(image)
        # ``segment_foot`` already selected the most foot-like contour and
        # returned the mask filled from it.  Re-deriving the contour from that
        # mask keeps the two in sync (the largest *candidate* contour is not
        # necessarily the selected one).
        selected = _collect_contours(mask)
        largest_contour = (
            max(selected, key=cv2.contourArea)
            if selected
            else max(contours, key=cv2.contourArea)
        )
        quality = footprint_quality(image, mask, largest_contour)
        if not quality.get("is_footprint", True):
            reasons = ", ".join(quality.get("reasons") or ["no footprint found"])
            raise ValueError(
                f"No footprint detected in the uploaded image ({reasons})."
            )
        roi, roi_mask, region_info = extract_roi(image, mask, largest_contour)

        result = {
            "roi": roi,
            "roi_mask": roi_mask,
            "mask": mask,
            "threshold": cleaned_threshold,
            "region_info": region_info,
            "quality": quality,
        }
        if return_visualization:
            result["visualization"] = visualize_regions(image, largest_contour, region_info)
        return result

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Error processing footprint: {error}") from error
