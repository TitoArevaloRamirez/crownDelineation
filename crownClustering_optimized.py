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
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import math

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
    normalize_per_band,
    compute_vegetation_indices,
    build_candidate_features,
    mask_bright_spots,
    rank_candidates,
    detect_big_round_blobs,
    filter_border_contours,
    top_k_blobs,
    get_blob_bounding_boxes,
    run_famnet,
    detect_strong_peaks,
    stack_feature_maps,
    compute_ground_removal_mask,
    grow_peak_circles_until_collision,
    sample_points_in_circle_xy,
    xy_to_rc,
    plot_grown_circles_debug,
    show_final_results,
)

from clustering_methods import (
    propagate_labels_random_walker,
    propagate_labels_sklearn_label_spreading,
    propagate_labels_gl_poisson,
    propagate_labels_gl_laplace,
    propagate_labels_gl_laplace_poisson_reweighted,
    propagate_labels_watershed,
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


def _dedupe_and_cap_seeds(
    points_rc: np.ndarray,
    max_seeds: int | None,
) -> np.ndarray:
    points_rc = np.asarray(points_rc)
    if points_rc.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    points_rc = np.asarray(points_rc, dtype=np.int64).reshape(-1, 2)
    points_rc = np.unique(points_rc, axis=0)
    if max_seeds is not None and len(points_rc) > max_seeds:
        # Deterministic, geometry-preserving sub-sampling: sort by row/col, then
        # pick evenly spaced points. This avoids random benchmark noise.
        order = np.lexsort((points_rc[:, 1], points_rc[:, 0]))
        points_rc = points_rc[order]
        keep = np.linspace(0, len(points_rc) - 1, int(max_seeds)).round().astype(int)
        points_rc = points_rc[keep]
    return points_rc


def make_disk_supports(
    image_shape: tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    radius: int | Sequence[int | float],
    valid_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Create simple per-label disk-union support masks around seed points.

    This is useful when benchmarking methods under the same crown-local prior as
    your random-walker implementation.
    """
    h, w = image_shape
    if isinstance(radius, Sequence) and not isinstance(radius, (str, bytes)):
        radii = list(radius)
        if len(radii) != len(seed_clusters_rc):
            raise ValueError("radius sequence must have one value per cluster")
    else:
        radii = [radius] * len(seed_clusters_rc)  # type: ignore[list-item]

    yy, xx = np.ogrid[:h, :w]
    supports: list[np.ndarray] = []
    base = (
        np.ones((h, w), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    for points, rad in zip(seed_clusters_rc, radii):
        mask = np.zeros((h, w), dtype=bool)
        r_float = float(rad)
        for r, c in _dedupe_and_cap_seeds(np.asarray(points), None):
            if 0 <= r < h and 0 <= c < w:
                mask |= (yy - int(r)) ** 2 + (xx - int(c)) ** 2 <= r_float**2
        supports.append(mask & base)
    return supports


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
    # data_np = np.stack((propagation_image, ranking_image), axis=-1)

    core, factors = tucker(propagation_image, rank=[h, w, 3], verbose=2)

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

    # plot_grown_circles_debug(
    #    image_rgb=X_hat[:, :, 0:3, 0],
    #    peaks_rc=peaks_rc,
    #    labels=circle_labels,
    #    circle_info=circle_info,
    #    debug=circle_debug,
    # )

    # labels = propagate_labels_random_walker(
    #    image=core[:, :, :, 0],  # X_hat[:, :, 0:5, 0],
    #    seed_clusters_rc=seed_clusters_rc,
    #    vegetation_mask=vegetation_mask,
    #    beta=args.rw_beta,
    #    neighborhood_radius=None,
    #    neighborhood_radii=valid_cluster_radii,
    #    max_seeds_per_cluster=20,
    #    enforce_label_neighborhoods=True,
    #    crop_to_active_bbox=True,
    #    use_probability_constraints=False,
    # )

    # labels = propagate_labels_sklearn_label_spreading(
    #    image=core[:, :, :, 0],  # X_hat[:, :, 0:5, 0],
    #    seed_clusters_rc=seed_clusters_rc,
    #    vegetation_mask=vegetation_mask,
    #    spatial_weight=0.25,
    #    gamma=20.0,
    #    alpha=0.2,
    #    max_iter=1000,
    #    neighborhood_radius=None,
    #    neighborhood_radii=valid_cluster_radii,
    #    max_seeds_per_cluster=20,
    # )

    # labels = propagate_labels_gl_poisson(
    #    image=core[:, :, :, 0],  # X_hat[:, :, 0:5, 0],
    #    seed_clusters_rc=seed_clusters_rc,
    #    vegetation_mask=vegetation_mask,
    #    max_iter=1000,
    #    max_seeds_per_cluster=20,
    #    neighborhood_radius=None,
    #    neighborhood_radii=valid_cluster_radii,
    #    enforce_neighborhoods=True,
    # )

    # labels = propagate_labels_gl_laplace(
    #    image=core[:, :, :],  # X_hat[:, :, 0:5, 0],
    #    seed_clusters_rc=seed_clusters_rc,
    #    vegetation_mask=vegetation_mask,
    #    max_seeds_per_cluster=20,
    #    neighborhood_radius=None,
    #    neighborhood_radii=valid_cluster_radii,
    #    enforce_neighborhoods=False,
    # )

    labels = propagate_labels_watershed(
        image=core[:, :, :],  # X_hat[:, :, 0:5, 0],
        seed_clusters_rc=seed_clusters_rc,
        vegetation_mask=vegetation_mask,
        max_seeds_per_cluster=20,
        neighborhood_radius=None,
        neighborhood_radii=valid_cluster_radii,
        enforce_label_neighborhoods=True,
    )

    # labels = propagate_labels_gl_laplace_poisson_reweighted(
    #   image=core[:, :, :, 0],  # X_hat[:, :, 0:5, 0],
    #   seed_clusters_rc=seed_clusters_rc,
    #   vegetation_mask=vegetation_mask,
    #   max_seeds_per_cluster=20,
    #   neighborhood_radius=None,
    #   neighborhood_radii=valid_cluster_radii,
    #   enforce_neighborhoods=True,
    # )

    show_final_results(
        cands_rank[best_name],
        X_hat[:, :, 0],  # density_smooth,
        X_hat[:, :, 0:3],
        labels,
        seed_points_rc,
    )


if __name__ == "__main__":
    main()
