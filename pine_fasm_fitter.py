"""
pine_fasm_fitter.py

Constrained FASM-style pine canopy fitting from:
    - a vegetation mask,
    - a user-provided top/apex anchor point,
    - a predefined/general pine canopy polygon prior.

This implementation follows the practical pipeline discussed for fitting a
standard pine canopy shape from above when the mask represents vegetation in
general, not necessarily the exact pine canopy.

Important note
--------------
This is a pragmatic, implementation-oriented FASM-style fitter. It does not
require a trained PCA/FASM statistical model. Instead, it uses:

    1. A sparse interpolation/control-point representation of the prior polygon.
    2. Per-segment signed "roughness" values, analogous to the FASM idea that
       boundary detail is represented compactly between interpolation points.
    3. Iterative curve attraction to the vegetation-mask boundary.
    4. Strong top-anchor and shape-prior regularization.

When all roughness values are zero, the generated curve becomes a piecewise
linear interpolation through the control points, matching the useful limiting
case discussed in FASM.

Dependencies
------------
Required:
    numpy
    scipy

Optional:
    none

Coordinate convention
---------------------
Image coordinates are assumed:
    x = column index
    y = row index

So a point is represented as:
    [x, y]

The vegetation mask is indexed as:
    mask[y, x]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


ArrayLike = Sequence[Sequence[float]]


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclass
class FittingParams:
    """Parameters controlling the constrained FASM-style fitting."""

    # Polygon / curve sampling
    n_curve_points: int = 256
    n_control_points: int = 25

    # Iteration
    max_iterations: int = 40
    convergence_threshold: float = 0.35

    # Normal search
    normal_search_radius: int = 12
    normal_search_step: float = 1.0

    # Mask preprocessing
    min_component_area: int = 20
    morphology_radius: int = 2
    fill_holes: bool = True

    # Fitting energy weights
    boundary_weight: float = 1.0
    outside_mask_weight: float = 25.0
    deformation_weight: float = 0.08
    prior_weight: float = 0.05
    top_anchor_weight: float = 1000.0

    # Shape update smoothing
    update_blend: float = 0.55
    roughness_limit: float = 0.25

    # Initialization search around the anchored prior.
    # Use [1.0] and [0.0] if you want no scale/rotation search.
    scale_candidates: Tuple[float, ...] = (0.85, 0.95, 1.0, 1.05, 1.15)
    rotation_candidates_deg: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)

    # If True, the top/apex point is fixed exactly at top_anchor.
    hard_top_anchor: bool = True


@dataclass
class FASMParameters:
    """
    Compact FASM-style representation.

    control_points:
        Closed polygon control/interpolation points, without duplicated final point.
        Shape: (K, 2)

    roughness:
        One signed roughness value per segment.
        Shape: (K,)

    top_control_index:
        Index of the control point corresponding to the anchored top/apex.
    """

    control_points: np.ndarray
    roughness: np.ndarray
    top_control_index: int = 0


@dataclass
class FitResult:
    """Result returned by fit_pine_canopy_fasm."""

    fitted_curve: np.ndarray
    fitted_polygon: np.ndarray
    fasm_parameters: FASMParameters
    score: Dict[str, float]
    iterations: int
    converged: bool
    history: Dict[str, list] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Basic geometry utilities
# ---------------------------------------------------------------------


def as_points(points: ArrayLike) -> np.ndarray:
    """Convert input to a float array of shape (N, 2)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected points with shape (N, 2), using [x, y] coordinates.")
    return arr


def ensure_open_polygon(points: ArrayLike, atol: float = 1e-9) -> np.ndarray:
    """
    Return a polygon without a duplicated final point.

    Many polygon formats repeat the first point at the end. Internally this
    module stores closed polygons without that duplicate point.
    """
    pts = as_points(points)
    if len(pts) < 3:
        raise ValueError("A polygon needs at least three points.")
    if np.linalg.norm(pts[0] - pts[-1]) <= atol:
        pts = pts[:-1]
    return pts


def close_polygon(points: ArrayLike) -> np.ndarray:
    """Return polygon with the first point duplicated at the end."""
    pts = ensure_open_polygon(points)
    return np.vstack([pts, pts[0]])


def polygon_arc_lengths(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Cumulative arc length for a closed polygon without duplicated endpoint."""
    pts = ensure_open_polygon(points)
    closed = close_polygon(pts)
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    return cumulative, float(cumulative[-1])


def resample_closed_polygon(points: ArrayLike, n_points: int) -> np.ndarray:
    """
    Uniformly resample a closed polygon boundary to n_points.

    Returns an open closed-boundary representation: shape (n_points, 2),
    without repeating the first point at the end.
    """
    pts = ensure_open_polygon(points)
    if n_points < 3:
        raise ValueError("n_points must be >= 3.")

    cumulative, total_length = polygon_arc_lengths(pts)
    if total_length <= 1e-12:
        raise ValueError("Degenerate polygon with near-zero perimeter.")

    closed = close_polygon(pts)
    targets = np.linspace(0.0, total_length, n_points, endpoint=False)
    out = np.zeros((n_points, 2), dtype=float)

    seg_idx = np.searchsorted(cumulative, targets, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(pts) - 1)

    seg_start_len = cumulative[seg_idx]
    seg_end_len = cumulative[seg_idx + 1]
    denom = np.maximum(seg_end_len - seg_start_len, 1e-12)
    t = (targets - seg_start_len) / denom

    out = (1.0 - t[:, None]) * closed[seg_idx] + t[:, None] * closed[seg_idx + 1]
    return out


def rotate_points_to_index(points: np.ndarray, index: int) -> np.ndarray:
    """Cyclically rotate a closed-boundary point sequence."""
    pts = np.asarray(points, dtype=float)
    index = int(index) % len(pts)
    return np.vstack([pts[index:], pts[:index]])


def nearest_point_index(points: np.ndarray, target: Sequence[float]) -> int:
    """Index of point nearest to target."""
    target_arr = np.asarray(target, dtype=float)
    d = np.linalg.norm(points - target_arr[None, :], axis=1)
    return int(np.argmin(d))


def default_top_index(points: np.ndarray) -> int:
    """
    Default apex/top point of a polygon in image coordinates.

    Since image y increases downward, the visually topmost point has minimum y.
    Ties are resolved by choosing the point closest to the median x.
    """
    pts = np.asarray(points, dtype=float)
    min_y = np.min(pts[:, 1])
    candidates = np.where(np.isclose(pts[:, 1], min_y))[0]
    if len(candidates) == 1:
        return int(candidates[0])
    median_x = np.median(pts[:, 0])
    best = candidates[np.argmin(np.abs(pts[candidates, 0] - median_x))]
    return int(best)


def transform_polygon_about_anchor(
    polygon: np.ndarray,
    source_anchor: Sequence[float],
    target_anchor: Sequence[float],
    scale: float = 1.0,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Scale and rotate polygon around source_anchor, then translate to target_anchor.
    """
    pts = ensure_open_polygon(polygon)
    source = np.asarray(source_anchor, dtype=float)
    target = np.asarray(target_anchor, dtype=float)

    theta = np.deg2rad(rotation_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)

    centered = pts - source[None, :]
    transformed = scale * (centered @ R.T) + target[None, :]
    return transformed


# ---------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------


def disk_structure(radius: int) -> np.ndarray:
    """Create a binary disk structuring element."""
    radius = int(max(0, radius))
    if radius == 0:
        return np.ones((1, 1), dtype=bool)

    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if min_area <= 1:
        return mask.astype(bool)

    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask.astype(bool)

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_area
    keep[0] = False
    return keep[labeled]


def preprocess_mask(mask: np.ndarray, params: Optional[FittingParams] = None) -> np.ndarray:
    """
    Clean a binary vegetation mask.

    Returns a boolean mask.
    """
    if params is None:
        params = FittingParams()

    m = np.asarray(mask).astype(bool)

    structure = disk_structure(params.morphology_radius)
    if params.morphology_radius > 0:
        m = ndimage.binary_opening(m, structure=structure)
        m = ndimage.binary_closing(m, structure=structure)

    if params.fill_holes:
        m = ndimage.binary_fill_holes(m)

    m = remove_small_components(m, params.min_component_area)
    return m.astype(bool)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel inner boundary from a binary mask."""
    m = np.asarray(mask).astype(bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)

    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return m & ~eroded


def inside_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Boolean mask indicating whether each [x, y] point lies inside the binary mask.
    Out-of-image points return False.
    """
    h, w = mask.shape
    pts = np.asarray(points, dtype=float)
    xi = np.rint(pts[:, 0]).astype(int)
    yi = np.rint(pts[:, 1]).astype(int)

    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(len(pts), dtype=bool)
    out[valid] = mask[yi[valid], xi[valid]]
    return out


def sample_distance_map(distance_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor sample of a distance map at [x, y] points.
    Out-of-image points receive a large penalty.
    """
    h, w = distance_map.shape
    pts = np.asarray(points, dtype=float)
    xi = np.rint(pts[:, 0]).astype(int)
    yi = np.rint(pts[:, 1]).astype(int)

    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(len(pts), fill_value=max(h, w), dtype=float)
    out[valid] = distance_map[yi[valid], xi[valid]]
    return out


# ---------------------------------------------------------------------
# FASM-style representation
# ---------------------------------------------------------------------


def select_control_points(
    curve_points: np.ndarray,
    n_control_points: int,
    top_anchor: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Select evenly spaced interpolation/control points from a dense closed curve.

    If top_anchor is given, the dense curve is cyclically rotated so index 0
    corresponds to the point nearest the top anchor. The first control point
    is therefore the top/apex point.
    """
    curve = np.asarray(curve_points, dtype=float)
    if len(curve) < 3:
        raise ValueError("Need at least three curve points.")
    if n_control_points < 3:
        raise ValueError("n_control_points must be >= 3.")
    if n_control_points > len(curve):
        raise ValueError("n_control_points cannot exceed number of curve points.")

    if top_anchor is not None:
        start = nearest_point_index(curve, top_anchor)
        curve = rotate_points_to_index(curve, start)

    idx = np.floor(np.linspace(0, len(curve), n_control_points, endpoint=False)).astype(int)
    controls = curve[idx].copy()
    top_control_index = 0
    return controls, top_control_index, idx


def estimate_segment_roughness(
    dense_curve: np.ndarray,
    control_indices: np.ndarray,
    control_points: np.ndarray,
    limit: float = 0.25,
) -> np.ndarray:
    """
    Estimate one signed roughness value per control segment.

    Roughness is measured as the mean signed normal deviation of dense points
    between two neighboring control points, normalized by segment length.

    This is a stable practical analogue of storing boundary detail between
    interpolation points.
    """
    curve = np.asarray(dense_curve, dtype=float)
    controls = np.asarray(control_points, dtype=float)
    idx = np.asarray(control_indices, dtype=int)
    k = len(controls)
    n = len(curve)

    roughness = np.zeros(k, dtype=float)

    for i in range(k):
        j0 = int(idx[i])
        j1 = int(idx[(i + 1) % k])

        if j1 <= j0:
            segment_points = np.vstack([curve[j0:], curve[:j1]])
        else:
            segment_points = curve[j0:j1]

        p0 = controls[i]
        p1 = controls[(i + 1) % k]
        v = p1 - p0
        length = float(np.linalg.norm(v))

        if length <= 1e-9 or len(segment_points) == 0:
            roughness[i] = 0.0
            continue

        tangent = v / length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        # Signed distances from line through p0-p1.
        signed = (segment_points - p0[None, :]) @ normal
        roughness[i] = float(np.mean(signed) / length)

    return np.clip(roughness, -abs(limit), abs(limit))


def compute_fasm_parameters_fcf(
    dense_points: ArrayLike,
    n_control_points: int = 25,
    top_anchor: Optional[Sequence[float]] = None,
    roughness_limit: float = 0.25,
) -> FASMParameters:
    """
    Compute simplified FASM-style parameters from a dense closed boundary.

    Name note:
        The paper uses Fractal Curve Fitting (FCF) for general 2D boundaries.
        This function provides the implementation role needed in the proposed
        pipeline: convert an ordered closed boundary into a compact interpolation
        representation plus per-segment detail parameters.
    """
    dense = ensure_open_polygon(dense_points)
    controls, top_idx, control_indices = select_control_points(
        dense,
        n_control_points=n_control_points,
        top_anchor=top_anchor,
    )

    # If we rotated the dense curve inside select_control_points, do the same
    # for roughness estimation.
    if top_anchor is not None:
        start = nearest_point_index(dense, top_anchor)
        dense_for_fit = rotate_points_to_index(dense, start)
    else:
        dense_for_fit = dense

    roughness = estimate_segment_roughness(
        dense_for_fit,
        control_indices,
        controls,
        limit=roughness_limit,
    )

    return FASMParameters(
        control_points=controls,
        roughness=roughness,
        top_control_index=top_idx,
    )


def generate_fractal_interpolation_curve(
    fasm: FASMParameters,
    n_points: int = 256,
) -> np.ndarray:
    """
    Generate a dense curve from simplified FASM-style parameters.

    Each control segment is interpolated as:
        linear segment + signed normal displacement

    The displacement has zero value at segment endpoints and maximum magnitude
    near the segment midpoint. With roughness=0, this becomes piecewise linear.

    Returns:
        Dense closed-boundary point sequence of shape (n_points, 2), without
        duplicated final point.
    """
    controls = ensure_open_polygon(fasm.control_points)
    roughness = np.asarray(fasm.roughness, dtype=float)
    k = len(controls)

    if len(roughness) != k:
        raise ValueError("roughness must have one value per control segment.")
    if n_points < k:
        raise ValueError("n_points must be >= number of control points.")

    # Allocate approximately equal samples to each segment.
    base = n_points // k
    remainder = n_points % k
    counts = np.full(k, base, dtype=int)
    counts[:remainder] += 1

    segments = []

    for i in range(k):
        p0 = controls[i]
        p1 = controls[(i + 1) % k]
        count = int(counts[i])

        if count <= 0:
            continue

        v = p1 - p0
        length = float(np.linalg.norm(v))

        if length <= 1e-9:
            seg = np.repeat(p0[None, :], count, axis=0)
            segments.append(seg)
            continue

        tangent = v / length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        t = np.linspace(0.0, 1.0, count, endpoint=False)
        linear = (1.0 - t[:, None]) * p0[None, :] + t[:, None] * p1[None, :]

        # Smooth endpoint-preserving displacement.
        displacement = roughness[i] * length * np.sin(np.pi * t)
        seg = linear + displacement[:, None] * normal[None, :]
        segments.append(seg)

    return np.vstack(segments)


# ---------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------


def estimate_normals(curve: np.ndarray) -> np.ndarray:
    """
    Estimate normals for a closed curve.

    Normal sign is not critical because fitting samples both directions.
    """
    pts = np.asarray(curve, dtype=float)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    tangent = next_pts - prev_pts

    norm = np.linalg.norm(tangent, axis=1)
    norm = np.maximum(norm, 1e-12)
    tangent = tangent / norm[:, None]

    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return normals


def fit_curve_to_mask_with_anchor(
    curve: np.ndarray,
    clean_mask: np.ndarray,
    top_anchor: Sequence[float],
    params: FittingParams,
    reference_curve: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Move curve points along their normals toward plausible vegetation-mask boundary.

    The energy favors:
        - being close to the vegetation boundary,
        - staying inside vegetation,
        - not moving too far from the current curve,
        - not moving too far from the original/prior reference curve,
        - keeping the top/apex anchored.
    """
    curve = np.asarray(curve, dtype=float)
    anchor = np.asarray(top_anchor, dtype=float)

    if reference_curve is None:
        reference_curve = curve
    else:
        reference_curve = np.asarray(reference_curve, dtype=float)
        if len(reference_curve) != len(curve):
            reference_curve = resample_closed_polygon(reference_curve, len(curve))

    boundary = extract_boundary(clean_mask)

    # Distance to nearest vegetation boundary.
    # boundary=True should have zero distance.
    if np.any(boundary):
        distance_to_boundary = ndimage.distance_transform_edt(~boundary)
    else:
        # If the mask is empty, fitting cannot use boundary evidence.
        distance_to_boundary = np.full(clean_mask.shape, fill_value=max(clean_mask.shape), dtype=float)

    normals = estimate_normals(curve)
    adjusted = curve.copy()

    top_index = nearest_point_index(curve, anchor)

    offsets = np.arange(
        -params.normal_search_radius,
        params.normal_search_radius + 0.5 * params.normal_search_step,
        params.normal_search_step,
        dtype=float,
    )

    h, w = clean_mask.shape

    for i, p in enumerate(curve):
        if params.hard_top_anchor and i == top_index:
            adjusted[i] = anchor
            continue

        candidates = p[None, :] + offsets[:, None] * normals[i][None, :]

        # Keep candidates within image bounds by penalizing outside points.
        xi = np.rint(candidates[:, 0]).astype(int)
        yi = np.rint(candidates[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

        boundary_cost = sample_distance_map(distance_to_boundary, candidates)
        is_inside = inside_mask(clean_mask, candidates)
        outside_cost = np.where(is_inside, 0.0, 1.0)

        move_cost = np.sum((candidates - p[None, :]) ** 2, axis=1)
        prior_cost = np.sum((candidates - reference_curve[i][None, :]) ** 2, axis=1)

        # Soft top anchor term for all points, strongest near the top index.
        # This prevents the upper canopy region from drifting to unrelated vegetation.
        circular_dist = min(abs(i - top_index), len(curve) - abs(i - top_index))
        top_band = np.exp(-(circular_dist ** 2) / (2.0 * (0.06 * len(curve)) ** 2))
        top_cost = top_band * np.sum((candidates - curve[i][None, :]) ** 2, axis=1)

        energy = (
            params.boundary_weight * boundary_cost
            + params.outside_mask_weight * outside_cost
            + params.deformation_weight * move_cost
            + params.prior_weight * prior_cost
            + params.top_anchor_weight * 0.001 * top_cost
        )

        # Strong penalty for candidates outside the image.
        energy = np.where(valid, energy, energy + 1e6)

        best = int(np.argmin(energy))
        adjusted[i] = candidates[best]

    if params.hard_top_anchor:
        adjusted[top_index] = anchor

    return adjusted


def blend_fasm_parameters(
    old: FASMParameters,
    new: FASMParameters,
    blend: float,
    roughness_limit: float,
) -> FASMParameters:
    """
    Blend old and new FASM parameters for stable iterative fitting.

    blend = 1.0 means fully use new parameters.
    blend = 0.0 means keep old parameters.
    """
    blend = float(np.clip(blend, 0.0, 1.0))

    if len(old.control_points) != len(new.control_points):
        raise ValueError("Cannot blend FASM parameters with different control counts.")

    controls = (1.0 - blend) * old.control_points + blend * new.control_points
    roughness = (1.0 - blend) * old.roughness + blend * new.roughness
    roughness = np.clip(roughness, -abs(roughness_limit), abs(roughness_limit))

    return FASMParameters(
        control_points=controls,
        roughness=roughness,
        top_control_index=old.top_control_index,
    )


# ---------------------------------------------------------------------
# Initialization and scoring
# ---------------------------------------------------------------------


def polygon_point_mask(
    polygon: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize a polygon into a boolean mask using a vectorized ray-casting test.

    This avoids requiring OpenCV, skimage, shapely, or matplotlib.
    """
    poly = ensure_open_polygon(polygon)
    h, w = image_shape

    out = np.zeros((h, w), dtype=bool)

    min_x = max(0, int(np.floor(np.min(poly[:, 0]))))
    max_x = min(w - 1, int(np.ceil(np.max(poly[:, 0]))))
    min_y = max(0, int(np.floor(np.min(poly[:, 1]))))
    max_y = min(h - 1, int(np.ceil(np.max(poly[:, 1]))))

    if min_x > max_x or min_y > max_y:
        return out

    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    x = xx.astype(float) + 0.5
    y = yy.astype(float) + 0.5

    inside = np.zeros_like(x, dtype=bool)

    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        intersects = ((ya > y) != (yb > y))
        denom = (yb - ya) if abs(yb - ya) > 1e-12 else 1e-12
        x_intersect = (xb - xa) * (y - ya) / denom + xa
        inside ^= intersects & (x < x_intersect)

    out[min_y : max_y + 1, min_x : max_x + 1] = inside
    return out


def score_polygon_against_mask(
    polygon: np.ndarray,
    clean_mask: np.ndarray,
    top_anchor: Sequence[float],
    prior_reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute diagnostic quality scores.

    Higher inside_fraction and overlap_iou are better.
    Lower mean_boundary_distance and anchor_error are better.
    """
    poly = ensure_open_polygon(polygon)
    boundary = extract_boundary(clean_mask)
    if np.any(boundary):
        distance_to_boundary = ndimage.distance_transform_edt(~boundary)
        mean_boundary_distance = float(np.mean(sample_distance_map(distance_to_boundary, poly)))
    else:
        mean_boundary_distance = float(max(clean_mask.shape))

    inside_fraction = float(np.mean(inside_mask(clean_mask, poly)))

    poly_mask = polygon_point_mask(poly, clean_mask.shape)
    intersection = np.logical_and(poly_mask, clean_mask).sum()
    union = np.logical_or(poly_mask, clean_mask).sum()
    overlap_iou = float(intersection / union) if union > 0 else 0.0

    anchor = np.asarray(top_anchor, dtype=float)
    anchor_error = float(np.min(np.linalg.norm(poly - anchor[None, :], axis=1)))

    score = {
        "inside_fraction": inside_fraction,
        "overlap_iou_with_vegetation_mask": overlap_iou,
        "mean_boundary_distance_px": mean_boundary_distance,
        "top_anchor_error_px": anchor_error,
    }

    if prior_reference is not None:
        prior = resample_closed_polygon(prior_reference, len(poly))
        score["mean_prior_deviation_px"] = float(np.mean(np.linalg.norm(poly - prior, axis=1)))

    return score


def initialize_shape_from_top_anchor(
    prior_polygon: ArrayLike,
    clean_mask: np.ndarray,
    top_anchor: Sequence[float],
    params: FittingParams,
    prior_top_index: Optional[int] = None,
) -> np.ndarray:
    """
    Anchor the general pine shape to the user-provided top point.

    A small scale/rotation search is performed by default. The candidate with
    best simple score against the vegetation mask is selected.
    """
    prior = ensure_open_polygon(prior_polygon)

    if prior_top_index is None:
        prior_top_index = default_top_index(prior)

    prior_top = prior[prior_top_index]
    anchor = np.asarray(top_anchor, dtype=float)

    best_polygon = None
    best_energy = np.inf

    for scale in params.scale_candidates:
        for rotation_deg in params.rotation_candidates_deg:
            candidate = transform_polygon_about_anchor(
                prior,
                source_anchor=prior_top,
                target_anchor=anchor,
                scale=scale,
                rotation_deg=rotation_deg,
            )

            # Use a quick boundary/inside score on dense samples.
            dense = resample_closed_polygon(candidate, params.n_curve_points)
            inside_fraction = np.mean(inside_mask(clean_mask, dense))

            boundary = extract_boundary(clean_mask)
            if np.any(boundary):
                dmap = ndimage.distance_transform_edt(~boundary)
                boundary_distance = np.mean(sample_distance_map(dmap, dense))
            else:
                boundary_distance = max(clean_mask.shape)

            # Lower is better.
            energy = (
                params.boundary_weight * boundary_distance
                + params.outside_mask_weight * (1.0 - inside_fraction)
            )

            if energy < best_energy:
                best_energy = float(energy)
                best_polygon = candidate

    if best_polygon is None:
        best_polygon = transform_polygon_about_anchor(
            prior,
            source_anchor=prior_top,
            target_anchor=anchor,
            scale=1.0,
            rotation_deg=0.0,
        )

    return best_polygon


# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------


def fit_pine_canopy_fasm(
    vegetation_mask: np.ndarray,
    top_anchor: Sequence[float],
    pine_general_shape: ArrayLike,
    params: Optional[FittingParams] = None,
    prior_top_index: Optional[int] = None,
    return_history: bool = True,
) -> FitResult:
    """
    Fit a general pine canopy polygon to a vegetation mask using a constrained
    FASM-style pipeline.

    Parameters
    ----------
    vegetation_mask:
        Binary image where 1/True means vegetation.

    top_anchor:
        User-provided canopy top/apex point in image coordinates [x, y].

    pine_general_shape:
        Prior pine canopy polygon, as points [x, y]. The polygon can be in image
        coordinates or a local coordinate system. Its top/apex will be aligned
        to top_anchor.

    params:
        Optional FittingParams instance.

    prior_top_index:
        Optional index of the apex/top point in pine_general_shape. If omitted,
        the point with minimum y is used.

    return_history:
        If True, store per-iteration diagnostics.

    Returns
    -------
    FitResult
        Contains fitted_curve, fitted_polygon, FASM parameters, score, iteration
        count, convergence flag, and optional history.
    """
    if params is None:
        params = FittingParams()

    if params.n_control_points >= params.n_curve_points:
        raise ValueError("n_control_points must be smaller than n_curve_points.")

    clean_mask = preprocess_mask(vegetation_mask, params)
    if not np.any(clean_mask):
        raise ValueError("The vegetation mask is empty after preprocessing.")

    # 1. Initialize prior polygon using the top anchor.
    initial_polygon = initialize_shape_from_top_anchor(
        prior_polygon=pine_general_shape,
        clean_mask=clean_mask,
        top_anchor=top_anchor,
        params=params,
        prior_top_index=prior_top_index,
    )

    initial_curve = resample_closed_polygon(initial_polygon, params.n_curve_points)
    prior_reference_curve = initial_curve.copy()

    # 2. Convert to simplified FASM parameters.
    fasm = compute_fasm_parameters_fcf(
        dense_points=initial_curve,
        n_control_points=params.n_control_points,
        top_anchor=top_anchor,
        roughness_limit=params.roughness_limit,
    )

    history = {
        "shape_change_px": [],
        "score": [],
    }

    converged = False
    last_curve = generate_fractal_interpolation_curve(fasm, params.n_curve_points)

    for iteration in range(1, params.max_iterations + 1):
        curve = generate_fractal_interpolation_curve(fasm, params.n_curve_points)

        adjusted_curve = fit_curve_to_mask_with_anchor(
            curve=curve,
            clean_mask=clean_mask,
            top_anchor=top_anchor,
            params=params,
            reference_curve=prior_reference_curve,
        )

        proposed_fasm = compute_fasm_parameters_fcf(
            dense_points=adjusted_curve,
            n_control_points=params.n_control_points,
            top_anchor=top_anchor,
            roughness_limit=params.roughness_limit,
        )

        fasm = blend_fasm_parameters(
            old=fasm,
            new=proposed_fasm,
            blend=params.update_blend,
            roughness_limit=params.roughness_limit,
        )

        new_curve = generate_fractal_interpolation_curve(fasm, params.n_curve_points)
        shape_change = float(np.mean(np.linalg.norm(new_curve - last_curve, axis=1)))

        if return_history:
            history["shape_change_px"].append(shape_change)
            history["score"].append(
                score_polygon_against_mask(
                    new_curve,
                    clean_mask,
                    top_anchor,
                    prior_reference=prior_reference_curve,
                )
            )

        if shape_change < params.convergence_threshold:
            converged = True
            last_curve = new_curve
            break

        last_curve = new_curve

    fitted_curve = generate_fractal_interpolation_curve(fasm, params.n_curve_points)
    fitted_polygon = close_polygon(fitted_curve)

    score = score_polygon_against_mask(
        fitted_curve,
        clean_mask,
        top_anchor,
        prior_reference=prior_reference_curve,
    )

    return FitResult(
        fitted_curve=fitted_curve,
        fitted_polygon=fitted_polygon,
        fasm_parameters=fasm,
        score=score,
        iterations=iteration,
        converged=converged,
        history=history if return_history else {},
    )


# ---------------------------------------------------------------------
# Optional small demo
# ---------------------------------------------------------------------


def _make_demo_prior() -> np.ndarray:
    """
    Create a simple teardrop-like pine canopy prior in local coordinates.

    The apex/top is near [0, 0], with image y increasing downward.
    """
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)

    # Wider near lower half, sharper near top.
    x = 35.0 * np.sin(angles) * (0.75 + 0.25 * np.cos(angles))
    y = 70.0 * (1.0 - np.cos(angles)) / 2.0

    # Move apex to exactly [0, 0].
    points = np.column_stack([x, y])
    points -= points[default_top_index(points)]
    return points


def _demo() -> None:
    """Run a tiny synthetic smoke test."""
    h, w = 180, 180
    top_anchor = np.array([90.0, 25.0])
    prior = _make_demo_prior()

    # Create a synthetic vegetation mask by rasterizing a transformed prior.
    true_shape = transform_polygon_about_anchor(
        prior,
        source_anchor=prior[default_top_index(prior)],
        target_anchor=top_anchor,
        scale=1.1,
        rotation_deg=8.0,
    )
    mask = polygon_point_mask(true_shape, (h, w))
    mask = ndimage.binary_dilation(mask, structure=disk_structure(3))

    params = FittingParams(
        n_curve_points=220,
        n_control_points=23,
        max_iterations=25,
        normal_search_radius=10,
    )

    result = fit_pine_canopy_fasm(
        vegetation_mask=mask,
        top_anchor=top_anchor,
        pine_general_shape=prior,
        params=params,
    )

    print("Converged:", result.converged)
    print("Iterations:", result.iterations)
    print("Score:", result.score)


if __name__ == "__main__":
    _demo()
