"""Self-occluded fruit contour segmentation from Liang et al. (2024).

This module implements the core of Section 2.4.1, "Self-occluded fruit
region contour segmentation", from:

    Liang et al., "Occlusion-aware fruit segmentation in complex natural
    environments under shape prior", Computers and Electronics in
    Agriculture, 217 (2024) 108620.

The paper's procedure is:
1. Start from an already extracted under-segmented fruit region.
2. Compute the centroid of the connected fruit mask.
3. Trace the fruit boundary in counter-clockwise order.
4. Build the distance sequence from each boundary point to the centroid.
5. Detect local minima in that distance signal.
6. Filter abnormal minima with a minimum horizontal-distance threshold.
7. Split the contour at the retained minima to obtain contour segments.

Assumptions in this implementation:
- The input is a single connected binary mask for one under-segmented
  self-occluded fruit region, after the paper's preliminary watershed step.
- The paper does not fully specify the exact outlier rejection rule for the
  local minima; here it is approximated with a circular minimum-separation
  filter derived from Eq. (7).
- The code returns contour segments and split points, which are the main
  outputs needed for the paper's later contour fitting stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import math

import numpy as np

Point = Tuple[int, int]


@dataclass(frozen=True)
class SelfOccludedSegmentationResult:
    """Outputs of the self-occluded contour segmentation stage."""

    centroid_xy: Tuple[float, float]
    contour_xy: np.ndarray
    distance_signal: np.ndarray
    smoothed_distance_signal: np.ndarray
    minima_indices: np.ndarray
    minima_points_xy: np.ndarray
    contour_segments_xy: List[np.ndarray]
    min_index_gap: int


def segment_self_occluded_fruit_contour(
    mask: np.ndarray,
    *,
    smoothing_window: int = 9,
    min_index_gap: int | None = None,
    max_minima: int | None = 2,
) -> SelfOccludedSegmentationResult:
    """Segment an under-segmented fruit contour at centroid-distance minima.

    Args:
        mask: Binary 2D mask. Non-zero values are treated as foreground.
        smoothing_window: Odd moving-average window applied on the circular
            centroid-distance signal before minimum detection.
        min_index_gap: Minimum contour-index separation between retained local
            minima. If None, a default value derived from Eq. (7) is used.
        max_minima: Maximum number of minima to retain. For a two-fruit overlap,
            the paper's examples imply two primary split points.

    Returns:
        A dataclass containing the ordered contour, centroid-distance signal,
        retained minima, and contour segments cut at those minima.
    """

    binary_mask = _normalize_mask(mask)
    if binary_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not np.any(binary_mask):
        raise ValueError("mask must contain at least one foreground pixel")

    contour_rc = _trace_ordered_contour(binary_mask)
    if contour_rc.shape[0] < 3:
        raise ValueError("unable to extract a valid contour from mask")

    centroid_rc = _mask_centroid_rc(binary_mask)
    distances = np.linalg.norm(contour_rc.astype(np.float64) - centroid_rc, axis=1)
    smoothed = _smooth_circular_signal(distances, smoothing_window)

    if min_index_gap is None:
        min_index_gap = _default_min_index_gap(smoothed)

    candidate_minima = _find_circular_local_minima(smoothed)
    filtered_minima = _filter_minima_by_separation(
        candidate_minima,
        smoothed,
        min_index_gap=min_index_gap,
        max_minima=max_minima,
    )

    contour_xy = contour_rc[:, ::-1]
    centroid_xy = (float(centroid_rc[1]), float(centroid_rc[0]))
    minima_points_xy = contour_xy[filtered_minima] if filtered_minima.size else np.empty((0, 2), dtype=int)
    contour_segments_xy = _split_contour_by_indices(contour_xy, filtered_minima)

    return SelfOccludedSegmentationResult(
        centroid_xy=centroid_xy,
        contour_xy=contour_xy,
        distance_signal=distances,
        smoothed_distance_signal=smoothed,
        minima_indices=filtered_minima,
        minima_points_xy=minima_points_xy,
        contour_segments_xy=contour_segments_xy,
        min_index_gap=min_index_gap,
    )


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == np.bool_:
        return array
    return array != 0


def _mask_centroid_rc(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("cannot compute centroid of an empty mask")
    return np.array([rows.mean(), cols.mean()], dtype=np.float64)


def _smooth_circular_signal(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        raise ValueError("smoothing_window must be odd")
    pad = window // 2
    extended = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(extended, kernel, mode="valid")


def _default_min_index_gap(smoothed_distances: np.ndarray) -> int:
    # Eq. (7) in the paper gives d_hmin = (pi / (2N)) * sum_i d_i.
    # The paper maps the distance signal to a plane and uses the horizontal
    # spacing between minima. We approximate that spacing directly in contour
    # samples, which works well when adjacent contour samples are about 1 pixel
    # apart.
    mean_distance = float(np.mean(smoothed_distances))
    return max(3, int(round((math.pi / 2.0) * mean_distance)))


def _find_circular_local_minima(values: np.ndarray) -> np.ndarray:
    minima: List[int] = []
    count = values.shape[0]
    for idx in range(count):
        prev_value = values[idx - 1]
        current_value = values[idx]
        next_value = values[(idx + 1) % count]
        is_minimum = (current_value <= prev_value and current_value < next_value) or (
            current_value < prev_value and current_value <= next_value
        )
        if is_minimum:
            minima.append(idx)
    return np.asarray(minima, dtype=int)


def _filter_minima_by_separation(
    minima_indices: np.ndarray,
    values: np.ndarray,
    *,
    min_index_gap: int,
    max_minima: int | None,
) -> np.ndarray:
    if minima_indices.size == 0:
        return minima_indices

    contour_length = values.shape[0]
    order = minima_indices[np.argsort(values[minima_indices])]
    selected: List[int] = []

    for idx in order.tolist():
        if any(_circular_index_distance(idx, kept, contour_length) < min_index_gap for kept in selected):
            continue
        selected.append(idx)
        if max_minima is not None and len(selected) >= max_minima:
            break

    return np.asarray(sorted(selected), dtype=int)


def _circular_index_distance(a: int, b: int, length: int) -> int:
    direct = abs(a - b)
    return min(direct, length - direct)


def _split_contour_by_indices(contour_xy: np.ndarray, split_indices: np.ndarray) -> List[np.ndarray]:
    if split_indices.size == 0:
        return [contour_xy.copy()]
    if split_indices.size == 1:
        idx = int(split_indices[0])
        return [np.concatenate([contour_xy[idx:], contour_xy[: idx + 1]], axis=0)]

    segments: List[np.ndarray] = []
    ordered = np.asarray(sorted(split_indices.tolist()), dtype=int)
    for start_idx, end_idx in zip(ordered, np.roll(ordered, -1)):
        if start_idx < end_idx:
            segment = contour_xy[start_idx : end_idx + 1]
        else:
            segment = np.concatenate([contour_xy[start_idx:], contour_xy[: end_idx + 1]], axis=0)
        segments.append(segment)
    return segments


def _trace_ordered_contour(mask: np.ndarray) -> np.ndarray:
    boundary = _boundary_pixels(mask)
    if not boundary.any():
        raise ValueError("mask contains no boundary pixels")

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    start_rc = _first_boundary_pixel(boundary)
    start_padded = (start_rc[0] + 1, start_rc[1] + 1)
    backtrack = (start_padded[0], start_padded[1] - 1)

    contour: List[Tuple[int, int]] = []
    current = start_padded
    initial_backtrack = backtrack
    max_steps = int(mask.size * 16)

    for _ in range(max_steps):
        contour.append((current[0] - 1, current[1] - 1))
        next_point, next_backtrack = _moore_next_boundary_pixel(padded, current, backtrack)
        if next_point is None:
            break
        if next_point == start_padded and next_backtrack == initial_backtrack:
            break
        current, backtrack = next_point, next_backtrack
    else:
        raise RuntimeError("boundary tracing did not terminate")

    contour_array = np.asarray(contour, dtype=int)
    contour_array = _remove_consecutive_duplicates(contour_array)
    if contour_array.shape[0] >= 2 and np.array_equal(contour_array[0], contour_array[-1]):
        contour_array = contour_array[:-1]
    return contour_array


def _remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep]


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    all_neighbors = np.logical_and.reduce([n.astype(bool) for n in neighbors])
    return center.astype(bool) & ~all_neighbors


def _first_boundary_pixel(boundary: np.ndarray) -> Tuple[int, int]:
    rows, cols = np.nonzero(boundary)
    order = np.lexsort((cols, rows))
    return int(rows[order[0]]), int(cols[order[0]])


def _moore_next_boundary_pixel(
    padded_mask: np.ndarray,
    current: Tuple[int, int],
    backtrack: Tuple[int, int],
) -> Tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
    neighbors = _neighbor_coordinates(current)
    start_index = neighbors.index(backtrack)

    for offset in range(1, 9):
        idx = (start_index + offset) % 8
        candidate = neighbors[idx]
        if padded_mask[candidate] != 0:
            new_backtrack = neighbors[(idx - 1) % 8]
            return candidate, new_backtrack
    return None, None


def _neighbor_coordinates(point: Tuple[int, int]) -> List[Tuple[int, int]]:
    row, col = point
    return [
        (row - 1, col - 1),
        (row - 1, col),
        (row - 1, col + 1),
        (row, col + 1),
        (row + 1, col + 1),
        (row + 1, col),
        (row + 1, col - 1),
        (row, col - 1),
    ]


def synthetic_overlapping_circles_mask(
    height: int = 180,
    width: int = 240,
    *,
    circles: Sequence[Tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Create a simple under-segmented overlap mask for testing."""

    if circles is None:
        circles = ((90.0, 88.0, 48.0), (90.0, 146.0, 48.0))

    yy, xx = np.mgrid[:height, :width]
    mask = np.zeros((height, width), dtype=bool)
    for center_y, center_x, radius in circles:
        mask |= (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    return mask


__all__ = [
    "SelfOccludedSegmentationResult",
    "segment_self_occluded_fruit_contour",
    "synthetic_overlapping_circles_mask",
]
