"""
Footprint Image Enhancement for Biometric Recognition.

This module enhances crease and ridge patterns in footprint images using
CLAHE (Contrast Limited Adaptive Histogram Equalization) followed by a
multi-orientation Gabor filter bank. The enhanced images improve feature
extraction and matching accuracy for biometric identification.
"""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: tuple = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast in the image, making subtle crease and ridge
    patterns more visible without amplifying noise excessively.

    Args:
        image: Grayscale input image (uint8).
        clip_limit: Threshold for contrast limiting. Higher values allow
            more contrast enhancement.
        tile_size: Size of the grid for histogram equalization. Smaller
            tiles provide more localized enhancement.

    Returns:
        CLAHE-enhanced grayscale image (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image)
    return enhanced


def create_gabor_kernel(
    theta_degrees: float,
    ksize: int = 21,
    sigma: float = 3.0,
    lambd: float = 8.0,
    gamma: float = 0.5,
) -> np.ndarray:
    """Create a single zero-mean Gabor kernel for a given orientation.

    The kernel is zero-meaned (DC bias removed) to ensure the filter
    responds only to texture and not to uniform brightness.

    Args:
        theta_degrees: Orientation of the Gabor filter in degrees.
        ksize: Size of the filter kernel (ksize x ksize).
        sigma: Standard deviation of the Gaussian envelope.
        lambd: Wavelength of the sinusoidal factor (in pixels).
        gamma: Spatial aspect ratio (ellipticity of the Gaussian).

    Returns:
        Zero-mean Gabor kernel as a numpy array (float64).
    """
    # Convert degrees to radians
    theta_rad = np.deg2rad(theta_degrees)

    # Create Gabor kernel using OpenCV
    kernel = cv2.getGaborKernel(
        ksize=(ksize, ksize),
        sigma=sigma,
        theta=theta_rad,
        lambd=lambd,
        gamma=gamma,
        psi=0,
        ktype=cv2.CV_64F,
    )

    # Zero-mean the kernel to eliminate DC bias
    kernel -= kernel.mean()

    return kernel


def apply_gabor_filter_bank(
    image: np.ndarray,
    num_orientations: int = 6,
    ksize: int = 21,
    sigma: float = 3.0,
    lambd: float = 8.0,
    gamma: float = 0.5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply a bank of Gabor filters at multiple orientations.

    Creates Gabor kernels at evenly spaced orientations and applies each
    to the input image. The results are fused using pixel-wise maximum
    to capture ridge/crease patterns regardless of their orientation.

    Args:
        image: Grayscale input image (uint8).
        num_orientations: Number of filter orientations. Default 6 produces
            filters at 0°, 30°, 60°, 90°, 120°, 150°.
        ksize: Size of each Gabor kernel.
        sigma: Standard deviation of the Gaussian envelope.
        lambd: Wavelength of the sinusoidal factor.
        gamma: Spatial aspect ratio.

    Returns:
        A tuple of:
            - max_response: Pixel-wise maximum across all orientations,
              normalized to 0-255 (uint8).
            - responses: List of individual filter responses (float64),
              one per orientation.
    """
    # Generate evenly spaced orientations in degrees
    orientations = [i * (180.0 / num_orientations) for i in range(num_orientations)]

    responses = []
    for theta_deg in orientations:
        # Create the Gabor kernel for this orientation
        kernel = create_gabor_kernel(
            theta_degrees=theta_deg,
            ksize=ksize,
            sigma=sigma,
            lambd=lambd,
            gamma=gamma,
        )

        # Apply the filter
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        responses.append(filtered)

    # Compute max-response fusion (pixel-wise maximum across all orientations)
    max_response = np.maximum.reduce(responses)

    # Normalize to 0-255 range
    max_response_normalized = cv2.normalize(
        max_response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return max_response_normalized, responses


def enhance_footprint(
    image: np.ndarray, apply_clahe_first: bool = True
) -> dict:
    """Main enhancement pipeline for footprint images.

    Applies CLAHE for local contrast enhancement followed by a Gabor
    filter bank to emphasize crease and ridge patterns at multiple
    orientations.

    Args:
        image: Input image, either BGR (3-channel) or grayscale (1-channel).
        apply_clahe_first: If True, apply CLAHE before the Gabor filter bank.
            This generally improves results by normalizing local contrast first.

    Returns:
        Dictionary with keys:
            - 'enhanced': Final enhanced image (uint8, grayscale).
            - 'clahe': CLAHE-enhanced image (uint8), or None if skipped.
            - 'gabor_responses': List of individual Gabor filter responses.
            - 'gabor_fused': Max-response Gabor fusion image (uint8).
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Optionally apply CLAHE
    clahe_result = None
    if apply_clahe_first:
        clahe_result = apply_clahe(gray)
        gabor_input = clahe_result
    else:
        gabor_input = gray

    # Apply Gabor filter bank
    gabor_fused, gabor_responses = apply_gabor_filter_bank(gabor_input)

    # Normalize the final result to 0-255
    enhanced = cv2.normalize(
        gabor_fused, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return {
        'enhanced': enhanced,
        'clahe': clahe_result,
        'gabor_responses': gabor_responses,
        'gabor_fused': gabor_fused,
    }


def get_enhancement_visualization(
    original: np.ndarray, enhancement_result: dict
) -> np.ndarray:
    """Create a side-by-side visualization grid of the enhancement stages.

    Produces a 2x2 grid showing:
        - Row 1: Original | CLAHE Enhanced
        - Row 2: Gabor Fused | Final Enhanced

    Each cell is 224x224 pixels with 2px white borders and text labels.

    Args:
        original: Original input image (BGR or grayscale).
        enhancement_result: Dictionary returned by enhance_footprint().

    Returns:
        Combined visualization image (BGR, uint8).
    """
    cell_size = 224
    border = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_color = (0, 255, 0)  # Green text

    def _prepare_cell(img, label):
        """Resize an image to cell_size x cell_size and add a label."""
        if img is None:
            # Create a black placeholder
            cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        else:
            # Ensure the image is resized to the target cell size
            resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

            # Convert grayscale to BGR for consistent display
            if len(resized.shape) == 2:
                cell = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            else:
                cell = resized.copy()

        # Add text label at the top-left corner
        cv2.putText(
            cell, label, (5, 20),
            font, font_scale, label_color, font_thickness,
        )
        return cell

    # Prepare each cell
    # Convert original to grayscale for consistent comparison
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original

    cell_original = _prepare_cell(original_gray, "Original")
    cell_clahe = _prepare_cell(enhancement_result.get('clahe'), "CLAHE Enhanced")
    cell_gabor = _prepare_cell(enhancement_result.get('gabor_fused'), "Gabor Fused")
    cell_enhanced = _prepare_cell(enhancement_result.get('enhanced'), "Final Enhanced")

    # Create 2px white border
    h_border = np.ones((cell_size, border, 3), dtype=np.uint8) * 255
    v_border = np.ones((border, cell_size * 2 + border, 3), dtype=np.uint8) * 255

    # Assemble Row 1: Original | CLAHE Enhanced
    row1 = np.hstack([cell_original, h_border, cell_clahe])

    # Assemble Row 2: Gabor Fused | Final Enhanced
    row2 = np.hstack([cell_gabor, h_border, cell_enhanced])

    # Combine rows with a horizontal border
    grid = np.vstack([row1, v_border, row2])

    return grid
