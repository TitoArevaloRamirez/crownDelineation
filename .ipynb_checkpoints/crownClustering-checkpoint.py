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
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tree crown detection pipeline")
    p.add_argument("--data-root", type=str, required=True, help="Directory containing multispectral TIFFs")
    p.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    p.add_argument("--model-path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth")
    p.add_argument("--gpu-id", type=int, default=0, help="GPU id; -1 = CPU")
    p.add_argument("--adapt", action="store_true", help="Run test-time adaptation")
    p.add_argument("--gradient-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-7)
    p.add_argument("--weight-mincount", type=float, default=1e-9)
    p.add_argument("--weight-perturbation", type=float, default=1e-4)
    p.add_argument("--day-for-ranking", type=int, default=1, choices=range(len(DATES)))
    p.add_argument("--day-for-propagation", type=int, default=0, choices=range(len(DATES)))
    p.add_argument("--crop", type=int, nargs=4, metavar=("ROW0", "ROW1", "COL0", "COL1"), default=(1024, 1544, 1024, 1544))
    p.add_argument("--top-k-blobs", type=int, default=5)
    p.add_argument("--num-exemplars", type=int, default=40)
    p.add_argument("--peak-min-distance", type=int, default=25)
    p.add_argument("--peak-percentile", type=float, default=90.0)
    p.add_argument("--neighborhood-radius", type=int, default=50)
    p.add_argument("--ground-method", type=str, default="combined", choices=["ndvi", "osavi", "msavi", "combined"])
    p.add_argument("--rw-beta", type=float, default=100.0)
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    resnet = Resnet50FPN().to(device).eval()
    regressor = CountRegressor(6, pool="mean").to(device).eval()

    state = torch.load(model_path, map_location=device)
    regressor.load_state_dict(state)
    return resnet, regressor


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def _load_band(root: str, band: str, date: str) -> np.ndarray:
    path = os.path.join(root, f"{band}_{date}.tif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing band file: {path}")
    return np.asarray(Image.open(path))


def read_data(root: str, dates: Sequence[str] = DATES, bands: Sequence[str] = FILE_BANDS) -> np.ndarray:
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


def normalize_per_band(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each (date, band) independently to [0, 1]."""
    x = np.asarray(data, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    for t in range(x.shape[0]):
        for b in range(x.shape[-1]):
            band = x[t, :, :, b]
            finite = np.isfinite(band)
            if not np.any(finite):
                continue
            lo = float(band[finite].min())
            hi = float(band[finite].max())
            if hi - lo > eps:
                out[t, :, :, b] = (band - lo) / (hi - lo)
    return out


def normalize_channels(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    for ch in range(image.shape[-1]):
        band = image[..., ch]
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        lo = float(band[finite].min())
        hi = float(band[finite].max())
        if hi - lo > eps:
            out[..., ch] = (band - lo) / (hi - lo)
        out[..., ch][~finite] = 0.0
    return out


def normalize_to_uint8(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)
    lo = float(img[finite].min())
    hi = float(img[finite].max())
    if hi - lo <= eps:
        return np.zeros(img.shape, dtype=np.uint8)
    out = np.zeros_like(img, dtype=np.float32)
    out[finite] = (img[finite] - lo) / (hi - lo)
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
        arr = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(arr)
        scaled = np.zeros_like(arr, dtype=np.float32)
        if np.any(finite):
            lo = float(arr[finite].min())
            hi = float(arr[finite].max())
            if hi > lo:
                scaled[finite] = (arr[finite] - lo) / (hi - lo)
        indices[name] = scaled

    return indices


def build_candidate_features(xn: np.ndarray, indices: Dict[str, np.ndarray], day: int) -> Dict[str, np.ndarray]:
    band_names = ("Blue", "Green", "Red", "RedEdge", "NIR")
    out = {name: xn[day, :, :, i] for i, name in enumerate(band_names)}
    out.update({k: v[day] for k, v in indices.items()})
    return out


# ---------------------------------------------------------------------------
# Feature ranking / blob filtering
# ---------------------------------------------------------------------------

def mask_bright_spots(candidates: Dict[str, np.ndarray], laplacian_ksize: int = 3, dilate_ksize: int = 5) -> Dict[str, np.ndarray]:
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


def compute_feature_metrics(feature_img: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    x = np.asarray(feature_img, dtype=np.float32)
    m = np.asarray(mask) > 0
    inside = x[m]
    outside = x[~m]
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    blob_areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int32)

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
    pooled_std = math.sqrt((std_in ** 2 + std_out ** 2) / 2.0)

    return {
        **base,
        "contrast_ratio": (mu_in - mu_out) / (abs(mu_out) + eps),
        "effect_size": (mu_in - mu_out) / (pooled_std + eps),
        "fisher_score": (mu_in - mu_out) ** 2 / (std_in ** 2 + std_out ** 2 + eps),
    }


def rank_candidates(candidates: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], sort_by: str = "contrast_ratio") -> List[Dict[str, float]]:
    rows = []
    for name, img in candidates.items():
        if name not in masks:
            continue
        row = compute_feature_metrics(img, masks[name])
        row["name"] = name
        rows.append(row)
    rows.sort(key=lambda x: x.get(sort_by, float("-inf")), reverse=True)
    return rows


def detect_big_round_blobs(mask: np.ndarray, min_area: float = 100.0, min_circularity: float = 0.6, min_solidity: float = 0.85) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    bw = (np.asarray(mask) > 0).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

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
        circularity = 4.0 * math.pi * area / (perimeter ** 2)
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


def filter_border_contours(shape: Tuple[int, int], contours: Sequence[np.ndarray], margin: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
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


def top_k_blobs(blob_info: Sequence[dict], contours: Sequence[np.ndarray], k: int) -> Tuple[List[dict], List[np.ndarray]]:
    if not blob_info:
        return [], []
    scores = np.array([b["area"] * b["circularity"] for b in blob_info], dtype=np.float32)
    idx = np.argsort(scores)[::-1][:k]
    return [blob_info[i] for i in idx], [contours[i] for i in idx]


def get_blob_bounding_boxes(contours: Sequence[np.ndarray]) -> Tuple[List[List[int]], float]:
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

def detect_strong_peaks(image: np.ndarray, min_distance: int = 5, percentile: float = 80.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    threshold = float(np.percentile(image, percentile))
    return peak_local_max(image, min_distance=min_distance, threshold_abs=threshold, exclude_border=True)


def sample_points_in_circle_xy(center_xy: Tuple[float, float], area: float, num_points: int, image_shape: Tuple[int, int], seed: int | None = None) -> np.ndarray:
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

def compute_ground_removal_mask(x: np.ndarray, method: str = "osavi", use_otsu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
    return (mask > 0).astype(np.uint8), idx


def build_cluster_neighborhood_masks(image_shape: Tuple[int, int], seed_clusters_rc: Sequence[np.ndarray], neighborhood_radius: int) -> np.ndarray:
    h, w = image_shape
    k = len(seed_clusters_rc)
    masks = np.zeros((k, h, w), dtype=np.uint8)
    if neighborhood_radius <= 0:
        masks[:] = 1
        return masks.astype(bool)
    for i, cluster in enumerate(seed_clusters_rc):
        for r, c in cluster:
            if 0 <= r < h and 0 <= c < w:
                cv2.circle(masks[i], (int(c), int(r)), neighborhood_radius, 1, thickness=-1)
    return masks.astype(bool)


def propagate_labels_random_walker(image: np.ndarray, seed_clusters_rc: Sequence[np.ndarray], beta: float = 100.0, use_ground_removal: bool = True, ground_method: str = "combined", neighborhood_radius: int = 50) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    h, w, c = image.shape
    if len(seed_clusters_rc) == 0:
        return np.zeros((h, w), dtype=np.int32)

    if use_ground_removal:
        if c < 5:
            raise ValueError("Ground removal requires first 5 channels = [Blue, Green, Red, RedEdge, NIR]")
        vegetation_mask, _ = compute_ground_removal_mask(image[..., :5], method=ground_method, use_otsu=True)
        eligible = vegetation_mask.astype(bool)
    else:
        eligible = np.ones((h, w), dtype=bool)

    neighborhoods = build_cluster_neighborhood_masks((h, w), seed_clusters_rc, neighborhood_radius)
    allowed = eligible & np.any(neighborhoods, axis=0) if neighborhood_radius > 0 else eligible.copy()

    markers = np.zeros((h, w), dtype=np.int32)
    markers[~allowed] = -1
    for cluster_id, cluster in enumerate(seed_clusters_rc, start=1):
        cluster_allowed = allowed & neighborhoods[cluster_id - 1]
        for r, c in cluster:
            if 0 <= r < h and 0 <= c < w and cluster_allowed[r, c]:
                markers[r, c] = cluster_id

    if np.count_nonzero(markers > 0) == 0:
        return np.zeros((h, w), dtype=np.int32)

    data = normalize_channels(image)
    data[~np.isfinite(data)] = 0.0

    try:
        rw = random_walker(data, markers, beta=beta, mode="cg_mg", channel_axis=-1, copy=True, return_full_prob=False)
    except Exception:
        rw = random_walker(data, markers, beta=beta, mode="cg", channel_axis=-1, copy=True, return_full_prob=False)

    labels = np.asarray(rw, dtype=np.int32)
    labels[markers == -1] = 0
    labels[~eligible] = 0

    if neighborhood_radius > 0:
        for cluster_id in range(1, len(seed_clusters_rc) + 1):
            invalid = (labels == cluster_id) & (~neighborhoods[cluster_id - 1])
            labels[invalid] = 0

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
        loss_perturb = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=use_gpu)
        loss = loss_count + loss_perturb
        if torch.is_tensor(loss):
            loss.backward()
            optimizer.step()
    regressor.eval()
    return regressor


def run_famnet(best_feature: np.ndarray, boxes: Sequence[Sequence[int]], resnet, regressor, args, device: torch.device) -> np.ndarray:
    if len(boxes) == 0:
        raise RuntimeError("No exemplar boxes found; cannot run FamNet.")

    cache_path = Path(args.output_dir) / f"density_day{args.day_for_ranking}_adapt{int(args.adapt)}.pt"
    if cache_path.exists():
        output = torch.load(cache_path, map_location=device)
        return np.asarray(format_for_plotting(output), dtype=np.float32)

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
    return np.asarray(format_for_plotting(output), dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def show_final_results(bestCand: np.ndarray, density_map: np.ndarray, rgb: np.ndarray, labels: np.ndarray, seed_points_rc: np.ndarray) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser()
    #os.makedirs(args.output_dir, exist_ok=True)
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
    _, contours = filter_border_contours(selected_mask.shape, top_k_blobs(blob_info, contours, args.top_k_blobs)[1])
    boxes, min_box_area = get_blob_bounding_boxes(contours)
    if len(boxes) == 0:
        raise RuntimeError("No valid exemplar blobs found after filtering.")

    density_map = run_famnet(cands_rank[best_name], boxes, resnet, regressor, args, device)
    density_smooth = ndi.gaussian_filter(density_map, sigma=3)
    peaks_rc = detect_strong_peaks(density_smooth, min_distance=args.peak_min_distance, percentile=args.peak_percentile)
    est_count = int(np.rint(density_map.sum()))
    print(f"Estimated crowns: {est_count} | detected peaks: {len(peaks_rc)}")
    if len(peaks_rc) == 0:
        raise RuntimeError("No peaks detected in density map.")

    feature_names = list(cands_prop.keys())
    propagation_image = np.concatenate([cands_prop[name][..., None] for name in feature_names], axis=-1)
    h, w = propagation_image.shape[:2]

    seed_clusters_rc: List[np.ndarray] = []
    all_seed_points_rc: List[np.ndarray] = []
    for row, col in peaks_rc:
        pts_xy = sample_points_in_circle_xy((float(col), float(row)), min_box_area, args.num_exemplars, (h, w))
        pts_rc = xy_to_rc(pts_xy)
        pts_rc[:, 0] = np.clip(pts_rc[:, 0], 0, h - 1)
        pts_rc[:, 1] = np.clip(pts_rc[:, 1], 0, w - 1)
        seed_clusters_rc.append(pts_rc)
        all_seed_points_rc.append(pts_rc)

    seed_points_rc = np.vstack(all_seed_points_rc) if all_seed_points_rc else np.empty((0, 2), dtype=np.int32)

    labels = propagate_labels_random_walker(
        image=propagation_image,
        seed_clusters_rc=seed_clusters_rc,
        beta=args.rw_beta,
        use_ground_removal= True,
        ground_method=args.ground_method,
        neighborhood_radius=args.neighborhood_radius,
    )

    print(labels)

    rgb = np.stack([cands_prop["Red"], cands_prop["Green"], cands_prop["Blue"]], axis=-1)
    show_final_results(cands_rank[best_name], density_smooth, rgb, labels, seed_points_rc)


if __name__ == "__main__":
    main()
