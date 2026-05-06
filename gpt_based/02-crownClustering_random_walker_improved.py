"""
Tree Crown Detection Pipeline (refactored)
=========================================
Main fixes applied:
- removes duplicate imports / duplicate function definitions
- fixes seed coordinate bug: sampling returns (x, y), propagation expects (row, col)
- removes hard-coded label==15 visualization bug
- removes hard-coded IMG_H/IMG_W reshapes; uses actual image shapes
- validates I/O and empty-intermediate cases
- makes caching deterministic and output-dir aware
- uses faster random-walker propagation by default instead of NetworkX Dijkstra
- keeps channel ordering explicit: [blue, green, red, red_edge, nir]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from tqdm import tqdm

import tensorly as tl

from tensorly.decomposition import tucker

from model import CountRegressor, Resnet50FPN
from utils import (
    MAPS,
    MincountLoss,
    PerturbationLoss,
    Scales,
    Transform,
    extract_features,
    format_for_plotting,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNELS = ("blue", "green", "red", "red_edge", "nir")
FILE_BANDS = ("b", "g", "r", "rEd", "nir")
DATES = ("2020_11_21_1", "2020_11_21_2", "2020_11_22_1")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tree crown detection pipeline")
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory containing multispectral TIFFs",
    )
    p.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory"
    )
    p.add_argument(
        "--model-path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth"
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU id; -1 = CPU")
    p.add_argument("--adapt", action="store_true", help="Run test-time adaptation")
    p.add_argument("--gradient-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-7)
    p.add_argument("--weight-mincount", type=float, default=1e-9)
    p.add_argument("--weight-perturbation", type=float, default=1e-4)
    p.add_argument("--day-for-ranking", type=int, default=1, choices=range(len(DATES)))
    p.add_argument(
        "--day-for-propagation", type=int, default=0, choices=range(len(DATES))
    )
    p.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("ROW0", "ROW1", "COL0", "COL1"),
        default=(720, 1000, 420, 720),
    )
    p.add_argument("--top-k-blobs", type=int, default=5)
    p.add_argument("--num-exemplars", type=int, default=30)
    p.add_argument("--peak-min-distance", type=int, default=25)
    p.add_argument("--peak-percentile", type=float, default=90.0)
    p.add_argument("--neighborhood-radius", type=int, default=50)
    p.add_argument(
        "--ground-method",
        type=str,
        default="combined",
        choices=["ndvi", "osavi", "msavi", "combined"],
    )
    p.add_argument("--rw-beta", type=float, default=600.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------


def get_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        print(f"===> Using GPU {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    print("===> Using CPU")
    return torch.device("cpu")


def load_models(model_path: str, device: torch.device):
    weights_path = Path(model_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    resnet = Resnet50FPN().to(device).eval()
    regressor = CountRegressor(6, pool="mean").to(device).eval()

    state = torch.load(weights_path, map_location=device)
    regressor.load_state_dict(state)
    return resnet, regressor


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


def _load_band(root: str, band: str, date: str) -> np.ndarray:
    path = Path(root) / f"{band}_{date}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing band file: {path}")
    return np.asarray(Image.open(path))


def read_data(
    root: str, dates: Sequence[str] = DATES, bands: Sequence[str] = FILE_BANDS
) -> np.ndarray:
    """Return array with shape (T, H, W, B)."""
    tensors = []
    for date in dates:
        planes = [_load_band(root, band, date) for band in bands]
        shapes = {arr.shape for arr in planes}
        if len(shapes) != 1:
            raise ValueError(f"Band shape mismatch for {date}: {sorted(shapes)}")
        tensors.append(np.stack(planes, axis=-1))
    return np.stack(tensors, axis=0)


def crop_data(data: np.ndarray, crop: Sequence[int]) -> np.ndarray:
    r0, r1, c0, c1 = map(int, crop)
    if not (0 <= r0 < r1 <= data.shape[1] and 0 <= c0 < c1 <= data.shape[2]):
        raise ValueError(f"Invalid crop {crop} for data shape {data.shape}")
    return data[:, r0:r1, c0:c1, :]


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def minmax_scale(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale one array to [0, 1] while preserving NaN handling."""
    x = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    lo = float(x[finite].min())
    hi = float(x[finite].max())
    if hi - lo <= eps:
        return out

    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def normalize_per_band(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each (date, band) independently to [0, 1]."""
    x = np.asarray(data, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    for t in range(x.shape[0]):
        for b in range(x.shape[-1]):
            out[t, :, :, b] = minmax_scale(x[t, :, :, b], eps=eps)
    return out


def normalize_channels(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    for ch in range(image.shape[-1]):
        out[..., ch] = minmax_scale(image[..., ch], eps=eps)
    return out


def normalize_to_uint8(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)
    out = minmax_scale(img, eps=eps)
    return np.uint8(np.clip(out, 0, 1) * 255)


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------


def compute_vegetation_indices(xn: np.ndarray) -> Dict[str, np.ndarray]:
    """Input shape: (T, H, W, 5). Output values scaled to [0, 1] per index."""
    xn = np.asarray(xn, dtype=np.float32)
    if xn.shape[-1] < 5:
        raise ValueError(f"Expected >=5 channels, got {xn.shape}")

    b, g, r, re, nir = (xn[..., i] for i in range(5))
    msavi_disc = np.maximum((2 * nir + 1) ** 2 - 8 * (nir - r), 0.0)

    indices = {
        "NDVI": safe_div(nir - r, nir + r),
        "SAVI": 1.5 * safe_div(nir - r, nir + r + 0.5),
        "OSAVI": 1.16 * safe_div(nir - r, nir + r + 0.16),
        "MSAVI": (2 * nir + 1 - np.sqrt(msavi_disc)) / 2.0,
        "DVI": nir - r,
        "NDRE": safe_div(nir - re, nir + re),
        "GNDVI": safe_div(nir - g, nir + g),
        "NRI": safe_div(r, r + g + b),
        "VARI": safe_div(g - r, g + r - b),
        "PPR": safe_div(g - b, g + b),
        "ARVI": safe_div(nir - (2 * r - b), nir + (2 * r - b)),
    }

    for name, arr in list(indices.items()):
        indices[name] = minmax_scale(arr)

    return indices


def build_candidate_features(
    xn: np.ndarray, indices: Dict[str, np.ndarray], day: int
) -> Dict[str, np.ndarray]:
    band_names = ("Blue", "Green", "Red", "RedEdge", "NIR")
    out = {name: xn[day, :, :, i] for i, name in enumerate(band_names)}
    out.update({k: v[day] for k, v in indices.items()})
    return out


def stack_feature_maps(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack feature maps in insertion order into an [H, W, C] cube."""
    return np.concatenate([features[name][..., None] for name in features], axis=-1)


# ---------------------------------------------------------------------------
# Feature ranking / blob filtering
# ---------------------------------------------------------------------------


def mask_bright_spots(
    candidates: Dict[str, np.ndarray], laplacian_ksize: int = 3, dilate_ksize: int = 5
) -> Dict[str, np.ndarray]:
    kernel = np.ones((dilate_ksize, dilate_ksize), dtype=np.uint8)
    masks: Dict[str, np.ndarray] = {}
    for name, img in candidates.items():
        gray = normalize_to_uint8(img)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=laplacian_ksize)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        dilated = cv2.dilate(lap, kernel, iterations=1)
        _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks[name] = thresh
    return masks


def compute_feature_metrics(
    feature_img: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> Dict[str, float]:
    x = np.asarray(feature_img, dtype=np.float32)
    m = np.asarray(mask) > 0
    inside = x[m]
    outside = x[~m]
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8
    )
    blob_areas = (
        stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int32)
    )

    base = {
        "bright_fraction": float(m.mean()),
        "n_blobs": int(len(blob_areas)),
        "mean_blob_area": float(blob_areas.mean()) if len(blob_areas) else 0.0,
        "max_blob_area": float(blob_areas.max()) if len(blob_areas) else 0.0,
    }
    if inside.size == 0 or outside.size == 0:
        return {**base, "contrast_ratio": 0.0, "effect_size": 0.0, "fisher_score": 0.0}

    mu_in, mu_out = float(inside.mean()), float(outside.mean())
    std_in, std_out = float(inside.std()), float(outside.std())
    pooled_std = math.sqrt((std_in**2 + std_out**2) / 2.0)

    return {
        **base,
        "contrast_ratio": (mu_in - mu_out) / (abs(mu_out) + eps),
        "effect_size": (mu_in - mu_out) / (pooled_std + eps),
        "fisher_score": (mu_in - mu_out) ** 2 / (std_in**2 + std_out**2 + eps),
    }


def rank_candidates(
    candidates: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    sort_by: str = "contrast_ratio",
) -> List[Dict[str, float]]:
    rows = []
    for name, img in candidates.items():
        if name not in masks:
            continue
        row = compute_feature_metrics(img, masks[name])
        row["name"] = name
        rows.append(row)
    rows.sort(key=lambda x: x.get(sort_by, float("-inf")), reverse=True)
    return rows


def detect_big_round_blobs(
    mask: np.ndarray,
    min_area: float = 100.0,
    min_circularity: float = 0.6,
    min_solidity: float = 0.85,
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    bw = (np.asarray(mask) > 0).astype(np.uint8) * 255
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_mask = np.zeros_like(bw)
    info: List[dict] = []
    kept: List[np.ndarray] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter**2)
        if circularity < min_circularity:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        kept.append(cnt)
        cv2.drawContours(kept_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        info.append({"area": area, "circularity": circularity, "solidity": solidity})

    return kept_mask, info, kept


def filter_border_contours(
    shape: Tuple[int, int], contours: Sequence[np.ndarray], margin: int = 1
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = shape
    kept = []
    out = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue
        kept.append(cnt)
    if kept:
        cv2.drawContours(out, kept, -1, 255, thickness=cv2.FILLED)
    return out, kept


def top_k_blobs(
    blob_info: Sequence[dict], contours: Sequence[np.ndarray], k: int
) -> Tuple[List[dict], List[np.ndarray]]:
    if not blob_info:
        return [], []
    scores = np.array(
        [b["area"] * b["circularity"] for b in blob_info], dtype=np.float32
    )
    idx = np.argsort(scores)[::-1][:k]
    return [blob_info[i] for i in idx], [contours[i] for i in idx]


def get_blob_bounding_boxes(
    contours: Sequence[np.ndarray],
) -> Tuple[List[List[int]], float]:
    if not contours:
        return [], 0.0
    boxes = []
    areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([y, x, y + h - 1, x + w - 1])
        areas.append(float(w * h))
    return boxes, float(min(areas))


# ---------------------------------------------------------------------------
# Seeding / peaks
# ---------------------------------------------------------------------------


def detect_strong_peaks(
    image: np.ndarray, min_distance: int = 5, percentile: float = 80.0
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    threshold = float(np.percentile(image, percentile))
    return peak_local_max(
        image, min_distance=min_distance, threshold_abs=threshold, exclude_border=True
    )


def sample_points_in_circle_xy(
    center_xy: Tuple[float, float],
    area: float,
    num_points: int,
    image_shape: Tuple[int, int],
    seed: int | None = None,
) -> np.ndarray:
    """Return points in (x, y) order."""
    rng = np.random.default_rng(seed)
    cx, cy = center_xy
    radius = math.sqrt(max(area, 1.0) / math.pi)
    h, w = image_shape

    pts_out = []
    while len(pts_out) < num_points:
        n = max(8, 2 * (num_points - len(pts_out)))
        r = radius * np.sqrt(rng.random(n))
        theta = 2 * math.pi * rng.random(n)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        pts = np.column_stack((xs[valid], ys[valid]))
        pts_out.extend(pts.tolist())
    return np.asarray(pts_out[:num_points], dtype=np.float32)


def xy_to_rc(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    return np.column_stack((points_xy[:, 1], points_xy[:, 0])).astype(np.int32)


# ---------------------------------------------------------------------------
# Ground removal / propagation
# ---------------------------------------------------------------------------


def compute_ground_removal_mask(
    x: np.ndarray,
    method: str = "osavi",
    use_otsu: bool = True,
    remove_shadow: bool = True,
    shadow_percentile: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 5:
        raise ValueError(f"Expected shape (H, W, >=5), got {x.shape}")

    _, _, r, re, nir = (x[..., i] for i in range(5))
    ndvi = safe_div(nir - r, nir + r)
    osavi = 1.16 * safe_div(nir - r, nir + r + 0.16)
    msavi_term = np.maximum((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - r), 0.0)
    msavi = (2.0 * nir + 1.0 - np.sqrt(msavi_term)) / 2.0
    ndre = safe_div(nir - re, nir + re)

    if method == "ndvi":
        idx = ndvi
    elif method == "osavi":
        idx = osavi
    elif method == "msavi":
        idx = msavi
    else:
        idx = 0.5 * ndvi + 0.3 * osavi + 0.2 * ndre

    idx8 = normalize_to_uint8(idx)
    if use_otsu:
        _, mask = cv2.threshold(idx8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (idx > 0.2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    vegetation_mask = (mask > 0).astype(np.uint8)

    if remove_shadow:
        vegetation_mask, _ = compute_shadow_removal_mask(
            x,
            vegetation_mask=vegetation_mask,
            shadow_percentile=shadow_percentile,
        )

    return vegetation_mask.astype(np.uint8), idx


def build_cluster_neighborhood_masks(
    image_shape: Tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    neighborhood_radius: int | None = None,
    neighborhood_radii: Sequence[int | float] | None = None,
) -> np.ndarray:
    """
    Build one allowed-neighborhood mask per seed cluster.

    Supports either:
      - one global radius: neighborhood_radius=50
      - one radius per cluster: neighborhood_radii=[r1, r2, ..., rk]

    Returns:
        masks: bool array [K, H, W]
    """
    h, w = image_shape
    k = len(seed_clusters_rc)

    masks = np.zeros((k, h, w), dtype=np.uint8)

    if k == 0:
        return masks.astype(bool)

    if neighborhood_radii is not None:
        radii = np.asarray(neighborhood_radii, dtype=np.float32)

        if len(radii) != k:
            raise ValueError(f"Expected {k} neighborhood radii, got {len(radii)}.")

    else:
        if neighborhood_radius is None:
            raise ValueError(
                "Either neighborhood_radius or neighborhood_radii must be provided."
            )

        radii = np.full(k, float(neighborhood_radius), dtype=np.float32)

    for i, cluster in enumerate(seed_clusters_rc):
        radius = int(round(radii[i]))

        if radius <= 0:
            masks[i, :, :] = 1
            continue

        for r, c in cluster:
            if 0 <= r < h and 0 <= c < w:
                cv2.circle(
                    masks[i],
                    (int(c), int(r)),
                    radius,
                    1,
                    thickness=-1,
                )

    return masks.astype(bool)


def propagate_labels_random_walker(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    beta: float = 100.0,
    neighborhood_radius: int | None = None,
    neighborhood_radii: Sequence[int | float] | None = None,
    max_seeds_per_cluster: int | None = 64,
    enforce_label_neighborhoods: bool = True,
    crop_to_active_bbox: bool = True,
    use_probability_constraints: bool = True,
) -> np.ndarray:
    """
    Random-walker crown propagation constrained by vegetation and crown-local masks.

    The propagation is intentionally conservative:
      1. Pixels outside the vegetation mask are forbidden.
      2. Pixels outside every crown neighborhood are forbidden.
      3. Optional connected-component pruning removes active regions that contain no seed.
      4. Optional per-label constraints prevent a crown label from occupying pixels
         outside that crown's own neighborhood.
      5. Optional active-domain cropping reduces the size of the random-walker solve.

    Args:
        image:
            Array [H, W, C]. Multispectral or feature image used by random walker.

        seed_clusters_rc:
            List of K arrays. Each array contains seed points for one crown,
            in (row, col) format.

        vegetation_mask:
            Boolean or uint8 array [H, W].
            1 / True  = valid vegetation pixel.
            0 / False = ground, shadow, background, or invalid pixel.

        beta:
            Random-walker beta parameter. Larger values make propagation more
            sensitive to feature differences and usually produce sharper borders.

        neighborhood_radius:
            Optional global radius for all clusters.

        neighborhood_radii:
            Optional independent radius per cluster. Recommended: use radii
            estimated by grow_peak_circles_until_collision(...).

        max_seeds_per_cluster:
            Optional cap on seed markers per crown. This reduces marker-count bias
            when one crown has many more sampled seed points than another. Set to
            None to use all valid seeds.

        enforce_label_neighborhoods:
            If True, a label can only be assigned inside its own neighborhood mask.
            This is stronger than only restricting the global active domain.

        crop_to_active_bbox:
            If True, run random_walker only on the bounding box containing active
            pixels. This can be much faster when neighborhoods occupy a small part
            of the image.

        use_probability_constraints:
            If True and enforce_label_neighborhoods is True, request full random-
            walker probabilities and select the best valid label per pixel after
            masking invalid label/pixel pairs. This avoids simply deleting labels
            after propagation and usually leaves fewer holes.

    Returns:
        labels:
            Integer array [H, W].
            0 = background / non-vegetation / forbidden.
            1..K = propagated crown labels.
    """
    image = np.asarray(image, dtype=np.float32)
    vegetation_mask = np.asarray(vegetation_mask).astype(bool)

    if image.ndim != 3:
        raise ValueError(f"Expected image shape [H, W, C], got {image.shape}")

    h, w, _ = image.shape

    if vegetation_mask.shape != (h, w):
        raise ValueError(
            f"vegetation_mask shape {vegetation_mask.shape} does not match image shape {(h, w)}"
        )

    n_clusters = len(seed_clusters_rc)
    if n_clusters == 0:
        return np.zeros((h, w), dtype=np.int32)

    neighborhoods = build_cluster_neighborhood_masks(
        image_shape=(h, w),
        seed_clusters_rc=seed_clusters_rc,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
    )

    if neighborhoods.shape != (n_clusters, h, w):
        raise ValueError(
            f"Expected neighborhoods shape {(n_clusters, h, w)}, got {neighborhoods.shape}"
        )

    # Global active domain: vegetation pixels that are inside at least one
    # crown-local neighborhood. This keeps the random walker from solving over
    # irrelevant background or distant canopy regions.
    allowed = vegetation_mask & np.any(neighborhoods, axis=0)
    if not np.any(allowed):
        print("Warning: no allowed propagation pixels after applying masks.")
        return np.zeros((h, w), dtype=np.int32)

    markers = np.zeros((h, w), dtype=np.int32)
    markers[~allowed] = -1

    def _prepare_cluster_seeds(cluster: np.ndarray, cluster_allowed: np.ndarray) -> np.ndarray:
        """Return valid, unique, optionally downsampled seed points for one crown."""
        pts = np.asarray(cluster, dtype=np.int32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = pts.reshape(-1, 2)

        in_bounds = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < h)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < w)
        )
        pts = pts[in_bounds]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        pts = pts[cluster_allowed[pts[:, 0], pts[:, 1]]]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        # Unique + sorted gives deterministic marker selection independent of
        # accidental duplicate samples.
        pts = np.unique(pts, axis=0)

        if max_seeds_per_cluster is not None and len(pts) > max_seeds_per_cluster:
            # Use a deterministic, spatially spread subset. Sorting by row/col
            # and sampling evenly avoids the bias of simply taking the first N.
            order = np.lexsort((pts[:, 1], pts[:, 0]))
            pts = pts[order]
            idx = np.linspace(0, len(pts) - 1, int(max_seeds_per_cluster), dtype=np.int32)
            pts = pts[idx]

        return pts.astype(np.int32, copy=False)

    valid_label_ids: List[int] = []

    for cluster_id, cluster in enumerate(seed_clusters_rc, start=1):
        cluster_allowed = vegetation_mask & neighborhoods[cluster_id - 1]
        pts = _prepare_cluster_seeds(cluster, cluster_allowed)
        if len(pts) == 0:
            continue

        markers[pts[:, 0], pts[:, 1]] = cluster_id
        valid_label_ids.append(cluster_id)

    if len(valid_label_ids) == 0:
        print("Warning: no valid seed points inside vegetation/neighborhood masks.")
        return np.zeros((h, w), dtype=np.int32)

    # Remove allowed connected components that contain no seed. This improves
    # numerical conditioning, avoids meaningless unlabeled islands, and reduces
    # the active solve size.
    structure = ndi.generate_binary_structure(2, 1)
    component_labels, _ = ndi.label(allowed, structure=structure)
    seeded_components = np.unique(component_labels[markers > 0])
    seeded_components = seeded_components[seeded_components > 0]

    if len(seeded_components) == 0:
        print("Warning: no seeded connected components in allowed mask.")
        return np.zeros((h, w), dtype=np.int32)

    reachable_allowed = np.isin(component_labels, seeded_components)
    markers[allowed & ~reachable_allowed] = -1
    allowed = allowed & reachable_allowed

    data = normalize_channels(image)
    data[~np.isfinite(data)] = 0.0

    if crop_to_active_bbox:
        active_rows, active_cols = np.where(allowed | (markers > 0))
        r0 = int(active_rows.min())
        r1 = int(active_rows.max()) + 1
        c0 = int(active_cols.min())
        c1 = int(active_cols.max()) + 1
    else:
        r0, r1, c0, c1 = 0, h, 0, w

    row_slice = slice(r0, r1)
    col_slice = slice(c0, c1)

    data_crop = data[row_slice, col_slice, :]
    markers_crop = markers[row_slice, col_slice]
    neighborhoods_crop = neighborhoods[:, row_slice, col_slice]
    vegetation_crop = vegetation_mask[row_slice, col_slice]

    positive_labels = np.unique(markers_crop[markers_crop > 0]).astype(np.int32)
    if len(positive_labels) == 0:
        print("Warning: no valid seed points after active-domain cropping.")
        return np.zeros((h, w), dtype=np.int32)

    want_probabilities = enforce_label_neighborhoods and use_probability_constraints

    def _run_random_walker(return_full_prob: bool):
        last_error = None
        for mode in ("cg_mg", "cg"):
            try:
                return random_walker(
                    data_crop,
                    markers_crop,
                    beta=beta,
                    mode=mode,
                    channel_axis=-1,
                    copy=True,
                    return_full_prob=return_full_prob,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError("random_walker failed in both cg_mg and cg modes") from last_error

    if want_probabilities:
        try:
            probabilities = np.asarray(_run_random_walker(return_full_prob=True), dtype=np.float32)

            # random_walker returns one probability plane per positive label,
            # ordered by sorted label id. With labels 1..K this aligns with
            # positive_labels, but we still keep the explicit mapping below.
            if probabilities.shape[0] != len(positive_labels):
                raise RuntimeError(
                    "Unexpected probability output shape from random_walker: "
                    f"{probabilities.shape}; expected first axis length {len(positive_labels)}."
                )

            label_neighborhoods = neighborhoods_crop[positive_labels - 1]
            label_allowed = label_neighborhoods & vegetation_crop[None, :, :]

            constrained_probabilities = probabilities.copy()
            constrained_probabilities[~label_allowed] = -np.inf

            has_candidate = np.any(label_allowed, axis=0) & (markers_crop != -1)
            labels_crop = np.zeros(markers_crop.shape, dtype=np.int32)

            if np.any(has_candidate):
                best_idx = np.argmax(constrained_probabilities[:, has_candidate], axis=0)
                labels_crop[has_candidate] = positive_labels[best_idx]

            # Preserve seed labels exactly. This protects against rare numerical
            # ties and documents the intended marker semantics.
            seed_pixels = markers_crop > 0
            labels_crop[seed_pixels] = markers_crop[seed_pixels]

        except Exception as exc:
            print(
                "Warning: probability-constrained random walker failed; "
                f"falling back to hard post-filtering. Details: {exc}"
            )
            rw = _run_random_walker(return_full_prob=False)
            labels_crop = np.asarray(rw, dtype=np.int32)
    else:
        rw = _run_random_walker(return_full_prob=False)
        labels_crop = np.asarray(rw, dtype=np.int32)

    # Universal cleanup.
    labels_crop[markers_crop == -1] = 0
    labels_crop[~vegetation_crop] = 0

    # Hard safety net: even when probability constraints are disabled or have
    # fallen back, no label is allowed outside its own crown neighborhood.
    if enforce_label_neighborhoods:
        for label_id in positive_labels:
            invalid = (labels_crop == label_id) & (~neighborhoods_crop[label_id - 1])
            labels_crop[invalid] = 0

    labels = np.zeros((h, w), dtype=np.int32)
    labels[row_slice, col_slice] = labels_crop
    return labels

# ---------------------------------------------------------------------------
# FamNet inference / adaptation
# ---------------------------------------------------------------------------


def adapt_regressor(regressor, features, boxes, args, device: torch.device):
    regressor.train()
    optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)
    use_gpu = device.type != "cpu"

    for step in tqdm(range(args.gradient_steps), desc="Adapting"):
        optimizer.zero_grad(set_to_none=True)
        output = regressor(features)
        loss_count = args.weight_mincount * MincountLoss(output, boxes, use_gpu=use_gpu)
        loss_perturb = args.weight_perturbation * PerturbationLoss(
            output, boxes, sigma=8, use_gpu=use_gpu
        )
        loss = loss_count + loss_perturb
        if torch.is_tensor(loss):
            loss.backward()
            optimizer.step()
    regressor.eval()
    return regressor


def run_famnet(
    best_feature: np.ndarray,
    boxes: Sequence[Sequence[int]],
    resnet,
    regressor,
    load_density,
    args,
    device: torch.device,
) -> np.ndarray:
    if len(boxes) == 0:
        raise RuntimeError("No exemplar boxes found; cannot run FamNet.")

    cache_path = (
        Path(args.output_dir)
        / f"density_day{args.day_for_ranking}_adapt{int(args.adapt)}.pt"
    )
    if load_density and cache_path.exists():
        output = torch.load(cache_path, map_location=device)

        out = format_for_plotting(output)
        if torch.is_tensor(out):
            out = out.detach().cpu().numpy()
        # return np.asarray(format_for_plotting(output), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    img_3ch = np.stack([best_feature, best_feature, best_feature], axis=-1)
    pil_img = Image.fromarray(np.uint8(np.clip(img_3ch, 0, 1) * 255))
    sample = Transform({"image": pil_img, "lines_boxes": [list(b) for b in boxes]})
    t_image = sample["image"].unsqueeze(0).to(device)
    t_boxes = sample["boxes"].unsqueeze(0).to(device)

    with torch.no_grad():
        features = extract_features(resnet, t_image, t_boxes, MAPS, Scales)

    if args.adapt:
        features.requires_grad_(True)
        regressor = adapt_regressor(regressor, features, t_boxes, args, device)

    with torch.no_grad():
        output = regressor(features).detach().cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cache_path)

    out = format_for_plotting(output)
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    return np.asarray(out, dtype=np.float32)

    # return np.asarray(format_for_plotting(output), dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def show_final_results(
    bestCand: np.ndarray,
    density_map: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    seed_points_rc: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(density_map, cmap="gray")
    if len(seed_points_rc):
        axes[0].scatter(seed_points_rc[:, 1], seed_points_rc[:, 0], s=2, c="r")
    axes[0].set_title("Density map + seed samples")
    axes[0].axis("off")

    axes[1].imshow(np.clip(rgb, 0, 1))
    axes[1].set_title("RGB")
    axes[1].axis("off")

    axes[2].imshow(np.clip(rgb, 0, 1))
    axes[2].imshow(labels, cmap="tab20", alpha=0.55)
    if len(seed_points_rc):
        axes[2].scatter(seed_points_rc[:, 1], seed_points_rc[:, 0], s=2, c="k")
    axes[2].set_title("Propagated crown regions")
    axes[2].axis("off")

    axes[3].imshow(np.clip(bestCand, 0, 1))
    axes[3].set_title("BestCand")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def draw_contour(
    contour, ax=None, closed=True, show_points=False, color="b", linewidth=2
):
    """
    Draw a contour using matplotlib.

    Parameters
    ----------
    contour : (N, 2) array-like
        Sequence of (x, y) points defining the contour.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure is created.
    closed : bool, default=True
        Whether to close the contour.
    show_points : bool, default=False
        Whether to plot the contour points.
    color : str, default='b'
        Line color.
    linewidth : float, default=2
        Line width.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the contour drawn.
    """
    contour = np.asarray(contour)

    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("Contour must be of shape (N, 2)")

    if ax is None:
        fig, ax = plt.subplots()

    x, y = contour[:, 0], contour[:, 1]

    # Close contour if needed
    if closed:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    ax.plot(x, y, color=color, linewidth=linewidth)

    if show_points:
        ax.scatter(contour[:, 0], contour[:, 1], color=color, s=10)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # useful for image coordinates

    return ax


def plot_grown_circles_debug(
    image_rgb,
    peaks_rc,
    labels,
    circle_info,
    debug,
    alpha=0.35,
):
    """
    Debug plot for grown circular regions.

    Args:
        image_rgb: [H, W, 3] image.
        peaks_rc: [K, 2] row/col peaks.
        labels: output labels from grow_peak_circles_until_collision.
        circle_info: output circle_info.
        debug: output debug dict.
    """
    peaks_rc = np.asarray(peaks_rc)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.clip(image_rgb, 0, 1))
    if len(peaks_rc):
        axes[0].scatter(peaks_rc[:, 1], peaks_rc[:, 0], s=15, c="red")
    axes[0].set_title("RGB + peaks")
    axes[0].axis("off")

    axes[1].imshow(debug["vegetation_mask"], cmap="gray")
    axes[1].set_title("Ground mask")
    axes[1].axis("off")

    axes[2].imshow(np.clip(image_rgb, 0, 1))
    axes[2].imshow(labels, cmap="tab20", alpha=alpha)
    axes[2].set_title("Final grown circles")
    axes[2].axis("off")

    axes[3].imshow(np.clip(image_rgb, 0, 1))
    for info in circle_info:
        r, c = info["center_rc"]
        radius = info["radius"]
        circ = plt.Circle(
            (c, r),
            radius,
            fill=False,
            linewidth=1.5,
        )
        axes[3].add_patch(circ)
        axes[3].text(
            c,
            r,
            f"{info['circle_id']}\nr={radius}\n{info['stop_reason']}",
            fontsize=7,
            ha="center",
            va="center",
        )
    axes[3].set_title("Circle radius + stop reason")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def make_circle_mask(shape, center_rc, radius):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    r, c = center_rc
    cv2.circle(mask, (int(c), int(r)), int(radius), 1, thickness=-1)
    return mask.astype(bool)


def compute_overlap_counts(circle_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Count how many circles cover each pixel."""
    if not circle_masks:
        raise ValueError("circle_masks must not be empty")
    stacked = np.stack(circle_masks, axis=0).astype(np.uint8)
    return stacked.sum(axis=0)


def grow_peak_circles_until_collision(
    peaks_rc,
    image=None,
    vegetation_mask=None,
    ground_method="combined",
    remove_shadow=True,
    shadow_percentile=15.0,
    max_intersection_frac=0.10,
    radius_step=2,
    initial_radius=2,
    max_radius=None,
    use_otsu=True,
):
    """
    Grow one circular region per peak using vegetation mask constraints.

    A circle stops growing if:
      1) non-vegetation intersection >= max_intersection_frac of its area
         where non-vegetation = ground + shadow + invalid pixels
      2) overlap with other circles >= max_intersection_frac of its area

    Either provide:
        vegetation_mask: [H, W], 1/True = valid vegetation

    Or provide:
        image: [H, W, >=5], channel order [Blue, Green, RedEdge, NIR]
        so the vegetation mask is computed internally.

    Returns:
        labels: int array [H, W], 0 background, 1..K circle ids
        circle_info: list of per-circle metadata
        debug: dict with masks and growth history
    """
    peaks_rc = np.asarray(peaks_rc, dtype=np.int32)

    if vegetation_mask is None:
        if image is None:
            raise ValueError("Either vegetation_mask or image must be provided.")

        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] < 5:
            raise ValueError(f"Expected image shape [H, W, >=5], got {image.shape}")

        vegetation_mask, vegetation_index = compute_ground_removal_mask(
            image[..., :5],
            method=ground_method,
            use_otsu=use_otsu,
            remove_shadow=remove_shadow,
            shadow_percentile=shadow_percentile,
        )
    else:
        vegetation_mask = np.asarray(vegetation_mask)
        vegetation_index = None

    vegetation_mask = vegetation_mask.astype(bool)
    h, w = vegetation_mask.shape
    k = len(peaks_rc)

    if k == 0:
        return (
            np.zeros((h, w), dtype=np.int32),
            [],
            {
                "vegetation_mask": vegetation_mask,
                "non_vegetation_mask": ~vegetation_mask,
                "vegetation_index": vegetation_index,
                "history": [],
            },
        )

    if max_radius is None:
        max_radius = int(np.hypot(h, w))

    if not (0.0 <= max_intersection_frac <= 1.0):
        raise ValueError("max_intersection_frac must be between 0 and 1.")

    non_vegetation_mask = ~vegetation_mask

    radii = np.full(k, int(initial_radius), dtype=np.int32)
    active = np.ones(k, dtype=bool)
    stop_reasons = ["active"] * k

    history = []

    while np.any(active):
        proposed_radii = radii.copy()
        proposed_radii[active] += int(radius_step)

        circle_masks = [
            make_circle_mask((h, w), peaks_rc[i], proposed_radii[i]) for i in range(k)
        ]
        overlap_counts = compute_overlap_counts(circle_masks)

        stop_now = np.zeros(k, dtype=bool)
        step_records = []

        for i in range(k):
            if not active[i]:
                continue

            circle_i = circle_masks[i]
            area_i = int(circle_i.sum())

            if area_i == 0:
                stop_now[i] = True
                stop_reasons[i] = "empty_circle"
                continue

            nonveg_intersection = int((circle_i & non_vegetation_mask).sum())
            nonveg_frac = nonveg_intersection / area_i

            overlap_intersection = int((circle_i & (overlap_counts > 1)).sum())
            overlap_frac = overlap_intersection / area_i

            reached_max_radius = proposed_radii[i] >= max_radius

            if nonveg_frac >= max_intersection_frac:
                stop_now[i] = True
                stop_reasons[i] = "non_vegetation_intersection"
            elif overlap_frac >= max_intersection_frac:
                stop_now[i] = True
                stop_reasons[i] = "circle_intersection"
            elif reached_max_radius:
                stop_now[i] = True
                stop_reasons[i] = "max_radius"

            step_records.append(
                {
                    "circle_id": i + 1,
                    "radius": int(proposed_radii[i]),
                    "area": area_i,
                    "nonveg_frac": float(nonveg_frac),
                    "overlap_frac": float(overlap_frac),
                    "stop": bool(stop_now[i]),
                    "reason": stop_reasons[i] if stop_now[i] else "active",
                }
            )

        # Accept growth only for circles that did not violate constraints.
        for i in range(k):
            if active[i] and not stop_now[i]:
                radii[i] = proposed_radii[i]

        active[stop_now] = False
        history.append(step_records)

        if np.all(proposed_radii >= max_radius):
            break

    final_masks = [make_circle_mask((h, w), peaks_rc[i], radii[i]) for i in range(k)]

    labels = np.zeros((h, w), dtype=np.int32)

    for i, mask in enumerate(final_masks, start=1):
        # Optional: keep labels only inside vegetation.
        valid_mask = mask & vegetation_mask

        # Avoid overwriting previous labels in overlaps.
        labels[valid_mask & (labels == 0)] = i

    final_overlap_counts = compute_overlap_counts(final_masks)
    circle_info = []

    for i in range(k):
        mask_i = final_masks[i]
        area_i = int(mask_i.sum())

        nonveg_frac = float((mask_i & non_vegetation_mask).sum() / max(area_i, 1))
        overlap_frac = float(
            (mask_i & (final_overlap_counts > 1)).sum() / max(area_i, 1)
        )

        circle_info.append(
            {
                "circle_id": i + 1,
                "center_rc": tuple(map(int, peaks_rc[i])),
                "radius": int(radii[i]),
                "area": area_i,
                "vegetated_area": int((mask_i & vegetation_mask).sum()),
                "nonveg_frac": nonveg_frac,
                "overlap_frac": overlap_frac,
                "stop_reason": stop_reasons[i],
            }
        )

    debug = {
        "vegetation_mask": vegetation_mask,
        "non_vegetation_mask": non_vegetation_mask,
        "vegetation_index": vegetation_index,
        "circle_masks": final_masks,
        "history": history,
    }

    return labels, circle_info, debug


def compute_shadow_removal_mask(
    x: np.ndarray,
    vegetation_mask: np.ndarray | None = None,
    shadow_percentile: float = 15.0,
    nir_percentile: float = 10.0,
    rededge_percentile: float = 10.0,
    use_morphology: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Remove shadow pixels using multispectral information.

    Input:
        x: [H, W, >=5], channel order [Blue, Green, Red, RedEdge, NIR]
        vegetation_mask: optional bool/uint8 mask from compute_ground_removal_mask
        shadow_percentile: low visible brightness percentile rejected as shadow
        nir_percentile: low NIR percentile rejected as shadow
        rededge_percentile: low RedEdge percentile rejected as shadow

    Returns:
        keep_mask: uint8 [H, W], 1 = vegetated and not shadow
        debug: dict with intermediate masks
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 5:
        raise ValueError(f"Expected shape (H, W, >=5), got {x.shape}")

    blue, green, red, red_edge, nir = (x[..., i] for i in range(5))

    visible_brightness = (blue + green + red) / 3.0
    multispec_brightness = (green + red_edge + nir) / 3.0

    finite = np.isfinite(visible_brightness) & np.isfinite(nir) & np.isfinite(red_edge)

    if vegetation_mask is not None:
        veg = vegetation_mask.astype(bool)
        valid = finite & veg
    else:
        veg = np.ones(x.shape[:2], dtype=bool)
        valid = finite

    if not np.any(valid):
        keep = np.zeros(x.shape[:2], dtype=np.uint8)
        return keep, {
            "visible_brightness": visible_brightness,
            "multispec_brightness": multispec_brightness,
            "shadow_mask": np.ones(x.shape[:2], dtype=bool),
            "vegetation_mask": veg,
        }

    vis_thr = float(np.percentile(visible_brightness[valid], shadow_percentile))
    nir_thr = float(np.percentile(nir[valid], nir_percentile))
    re_thr = float(np.percentile(red_edge[valid], rededge_percentile))

    shadow_mask = (
        (visible_brightness <= vis_thr)
        & (nir <= nir_thr)
        & (red_edge <= re_thr)
        & valid
    )

    keep_mask = veg & finite & (~shadow_mask)

    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep_u8 = keep_mask.astype(np.uint8)
        keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_OPEN, kernel)
        keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, kernel)
        keep_mask = keep_u8.astype(bool)

    debug = {
        "visible_brightness": visible_brightness,
        "multispec_brightness": multispec_brightness,
        "shadow_mask": shadow_mask,
        "vegetation_mask": veg,
        "vis_thr": vis_thr,
        "nir_thr": nir_thr,
        "rededge_thr": re_thr,
    }

    return keep_mask.astype(np.uint8), debug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = get_device(args.gpu_id)
    resnet, regressor = load_models(args.model_path, device)

    data = read_data(args.data_root)
    data = crop_data(data, args.crop)
    xn = normalize_per_band(data)

    veg_indices = compute_vegetation_indices(xn)
    cands_rank = build_candidate_features(xn, veg_indices, day=args.day_for_ranking)
    cands_prop = build_candidate_features(xn, veg_indices, day=args.day_for_propagation)

    masks = mask_bright_spots(cands_rank)
    ranked = rank_candidates(cands_rank, masks, sort_by="contrast_ratio")
    if not ranked:
        raise RuntimeError("No candidate features could be ranked.")

    best_name = ranked[0]["name"]
    print(f"Best feature: {best_name}")

    selected_mask, blob_info, contours = detect_big_round_blobs(masks[best_name])
    _, contours = filter_border_contours(
        selected_mask.shape, top_k_blobs(blob_info, contours, args.top_k_blobs)[1]
    )
    boxes, min_box_area = get_blob_bounding_boxes(contours)
    if len(boxes) == 0:
        raise RuntimeError("No valid exemplar blobs found after filtering.")

    print(len(boxes))

    # Realmente necesitamos famnet?
    density_map = run_famnet(
        cands_rank[best_name], boxes, resnet, regressor, False, args, device
    )
    density_smooth = ndi.gaussian_filter(density_map, sigma=3)
    peaks_rc = detect_strong_peaks(
        density_smooth,
        min_distance=args.peak_min_distance,
        percentile=args.peak_percentile,
    )
    est_count = int(np.rint(density_map.sum()))
    print(f"Estimated crowns: {est_count} | detected peaks: {len(peaks_rc)}")
    if len(peaks_rc) == 0:
        raise RuntimeError("No peaks detected in density map.")

    propagation_image = stack_feature_maps(cands_prop)
    ranking_image = stack_feature_maps(cands_rank)
    h, w = propagation_image.shape[:2]

    vegetation_mask, _ = compute_ground_removal_mask(
        propagation_image[:, :, 0:5],
        method=args.ground_method,
        use_otsu=True,
        remove_shadow=True,
    )
    # plt.figure()
    # plt.imshow(vegetation_mask)
    # plt.figure()
    # plt.imshow(propagation_image[:, :, 0:3])
    # plt.show()
    # os.exit()

    circle_labels, circle_info, circle_debug = grow_peak_circles_until_collision(
        peaks_rc=peaks_rc,
        vegetation_mask=vegetation_mask,
        max_intersection_frac=0.15,
        radius_step=2,
        initial_radius=3,
    )

    cluster_radii = np.asarray(
        [info["radius"] for info in circle_info],
        dtype=np.float32,
    )

    # Build the decomposition input from two distinct days instead of duplicating
    # the same feature cube twice.
    data_np = np.stack((propagation_image, ranking_image), axis=-1)

    core, factors = tucker(data_np, rank=[h, w, 3, 1], verbose=2)

    X_hat = tl.tucker_to_tensor((core, factors))  # reconstructed tensor

    seed_clusters_rc: List[np.ndarray] = []
    all_seed_points_rc: List[np.ndarray] = []
    valid_cluster_radii: List[float] = []
    valid_peaks_rc: List[Tuple[int, int]] = []

    for i, (row, col) in enumerate(peaks_rc):
        pts_xy = sample_points_in_circle_xy(
            (float(col), float(row)),
            min_box_area - 5,
            args.num_exemplars,
            (h, w),
        )

        pts_rc = xy_to_rc(pts_xy)
        pts_rc[:, 0] = np.clip(pts_rc[:, 0], 0, h - 1)
        pts_rc[:, 1] = np.clip(pts_rc[:, 1], 0, w - 1)

        keep = vegetation_mask[pts_rc[:, 0], pts_rc[:, 1]] > 0
        pts_rc = pts_rc[keep]

        if len(pts_rc) == 0:
            continue

        seed_clusters_rc.append(pts_rc)
        all_seed_points_rc.append(pts_rc)
        valid_cluster_radii.append(float(cluster_radii[i]))
        valid_peaks_rc.append((int(row), int(col)))

    seed_points_rc = (
        np.vstack(all_seed_points_rc)
        if all_seed_points_rc
        else np.empty((0, 2), dtype=np.int32)
    )

    valid_cluster_radii = np.asarray(valid_cluster_radii, dtype=np.float32)
    valid_peaks_rc = np.asarray(valid_peaks_rc, dtype=np.int32)

    plot_grown_circles_debug(
        image_rgb=X_hat[:, :, 0:3, 0],
        peaks_rc=peaks_rc,
        labels=circle_labels,
        circle_info=circle_info,
        debug=circle_debug,
    )

    labels = propagate_labels_random_walker(
        image=X_hat[:, :, 0:5, 0],
        seed_clusters_rc=seed_clusters_rc,
        vegetation_mask=vegetation_mask,
        beta=args.rw_beta,
        neighborhood_radius=None,
        neighborhood_radii=valid_cluster_radii,
    )

    # for i_label in labels_unique:
    #    if i_label == 0:
    #        continue

    #    mask = np.zeros((h, w))
    #    mask[labels == i_label] = 255

    #    result = segment_self_occluded_fruit_contour(mask)

    #    fig, ax = plt.subplots()
    #    ax.imshow(X_hat[:, :, 0:3, 0])

    #    for segment in result.contour_segments_xy:

    #        print(segment.shape)
    #        ax = draw_contour(segment, ax, show_points=True)
    #    #print(result)

    #    plt.show()

    #    print(i_label.shape)
    #    print(i_label)
    #    #os.exit()

    # rgb = np.stack(
    #    [cands_prop["Red"], cands_prop["Green"], cands_prop["Blue"]], axis=-1
    # )
    show_final_results(
        cands_rank[best_name],
        X_hat[:, :, 0, 0],  # density_smooth,
        X_hat[:, :, 0:3, 0],
        labels,
        seed_points_rc,
    )


if __name__ == "__main__":
    main()
