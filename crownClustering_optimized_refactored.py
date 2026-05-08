"""Tree crown clustering pipeline.

Supports two acquisition modes:
1. multispectral stacks from separate band/date files;
2. a single RGB image with optional manual exemplar boxes.

The pipeline is designed to produce conservative, high-quality initial crown
clusters and crown bounding boxes for downstream deep-learning fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import matplotlib.pylab as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters


from clustering_methods_refactored import (
        propagate_labels_gl_laplace,
        propagate_labels_gl_laplace_poisson_reweighted,
        propagate_labels_gl_poisson,
        propagate_labels_random_walker,
        propagate_labels_sklearn_label_spreading,
        propagate_labels_watershed,
    )

from utils_refactored import (
    DATES,
    FILE_BANDS,
    detect_closed_boundary_boxes,
    adjust_boxes_for_crop,
    box_radius_estimates,
    boxes_to_centers_rc,
    build_candidate_features,
    compute_scene_vegetation_mask,
    detect_big_round_blobs,
    detect_strong_peaks,
    filter_border_contours,
    get_blob_bounding_boxes,
    grow_peak_circles_until_collision,
    labels_to_bounding_boxes,
    mask_bright_spots,
    minmax_scale,
    normalize_per_band,
    plot_grown_circles_debug,
    postprocess_crown_labels,
    rank_candidates,
    read_bbox_file,
    reduce_feature_cube,
    run_famnet,
    sample_seed_clusters_from_peaks,
    save_label_overlay,
    select_or_load_exemplar_boxes,
    show_final_results,
    stack_feature_maps,
    top_k_blobs,
    validate_bounding_boxes,
    write_bbox_file,
    save_clustered_tree_crown_patches,
    )


CLUSTERING_METHODS = (
    "watershed",
    "random_walker",
    "label_spreading",
    "gl_poisson",
    "gl_laplace",
    "gl_laplace_poisson_reweighted",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tree crown canopy delineation pipeline")

    p.add_argument(
        "--input-mode",
        type=str,
        default="multispectral",
        choices=["multispectral", "rgb"],
        help="Input type: multispectral stack from band files or a single RGB image.",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Directory containing multispectral TIFFs named like b_2020_11_21_1.tif.",
    )
    p.add_argument(
        "--rgb-image-path",
        type=str,
        default=None,
        help="Path to a single RGB image.",
    )
    p.add_argument("--output-dir", type=str, default="./output", help="Output directory.")
    p.add_argument(
        "--model-path",
        type=str,
        default="./data/pretrainedModels/FamNet_Save1.pth",
        help="Path to FamNet regressor weights. Not required with --seed-source manual_centers.",
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU id; -1 = CPU.")
    p.add_argument("--adapt", action="store_true", help="Run FamNet test-time adaptation.")
    p.add_argument("--gradient-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-7)
    p.add_argument("--weight-mincount", type=float, default=1e-9)
    p.add_argument("--weight-perturbation", type=float, default=1e-4)

    p.add_argument("--dates", type=str, nargs="*", default=list(DATES), help="Multispectral date suffixes.")
    p.add_argument("--bands", type=str, nargs="*", default=list(FILE_BANDS), help="Multispectral band prefixes.")
    p.add_argument("--day-for-ranking", type=int, default=1, help="Temporal slice used for exemplar ranking.")
    p.add_argument("--day-for-propagation", type=int, default=0, help="Temporal slice used for propagation features.")
    p.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("ROW0", "ROW1", "COL0", "COL1"),
        default=None,
        help="Optional crop in full-image coordinates. Default: no crop.",
    )

    p.add_argument(
        "--bbox-file",
        type=str,
        default=None,
        help="Optional manual exemplar/best-guess boxes file with one 'y1 x1 y2 x2' box per line.",
    )
    p.add_argument(
        "--interactive-bboxes",
        action="store_true",
        help="Draw exemplar boxes interactively, similar to demo.py. Useful for RGB images.",
    )
    p.add_argument(
        "--bbox-coordinate-space",
        choices=["full", "crop"],
        default="full",
        help="Interpret --bbox-file coordinates as full-image or already-cropped coordinates.",
    )
    p.add_argument(
        "--seed-source",
        choices=["density_peaks", "manual_centers"],
        default="density_peaks",
        help="Use FamNet density peaks or the centers of manual boxes as crown seeds.",
    )
    p.add_argument(
        "--fallback-to-manual-centers",
        action="store_true",
        help="If density peak detection fails and manual boxes exist, use their centers.",
    )

    p.add_argument("--top-k-blobs", type=int, default=5, help="Auto exemplar count.")
    p.add_argument("--auto-min-blob-area", type=float, default=50.0)
    p.add_argument("--auto-min-circularity", type=float, default=0.6)
    p.add_argument("--auto-min-solidity", type=float, default=0.85)
    p.add_argument("--num-exemplars", type=int, default=30, help="Seed samples per crown.")
    p.add_argument("--peak-min-distance", type=int, default=25)
    p.add_argument("--peak-percentile", type=float, default=90.0)
    p.add_argument("--default-radius", type=float, default=35.0, help="Fallback crown support radius.")
    p.add_argument("--manual-box-radius-scale", type=float, default=0.75)
    p.add_argument("--neighborhood-scale", type=float, default=1.15)
    p.add_argument("--max-intersection-frac", type=float, default=0.15)
    p.add_argument("--circle-radius-step", type=int, default=2)
    p.add_argument("--circle-initial-radius", type=int, default=3)

    p.add_argument(
        "--ground-method",
        type=str,
        default="combined",
        choices=["ndvi", "osavi", "msavi", "combined"],
    )
    p.add_argument(
        "--vegetation-mask-mode",
        type=str,
        default="auto",
        choices=["auto", "all"],
        help="Use an automatic vegetation mask or allow all finite pixels.",
    )
    p.add_argument("--disable-shadow-removal", action="store_true")

    p.add_argument("--rw-beta", type=float, default=600.0)
    p.add_argument("--watershed-gradient-sigma", type=float, default=1.0)
    p.add_argument("--watershed-distance-weight", type=float, default=0.15)
    p.add_argument("--watershed-compactness", type=float, default=0.0)
    p.add_argument(
        "--clustering-method",
        type=str,
        default="watershed",
        choices=CLUSTERING_METHODS,
    )
    p.add_argument(
        "--feature-reduction",
        choices=["pca", "none", "tucker"],
        default="tucker",
        help="Channel reduction before clustering. PCA is faster than full Tucker.",
    )
    p.add_argument("--feature-components", type=int, default=3)
    p.add_argument("--max-seeds-per-cluster", type=int, default=20)
    p.add_argument("--min-label-area", type=int, default=20)
    p.add_argument("--keep-largest-component", action="store_true", default=True)
    p.add_argument("--no-keep-largest-component", dest="keep_largest_component", action="store_false")
    p.add_argument("--no-fill-holes", dest="fill_holes", action="store_false", default=True)

    p.add_argument("--load-density", action="store_true", help="Reuse cached density map if available.")
    p.add_argument("--save-debug", action="store_true", help="Save circle-growth debug plot.")
    p.add_argument("--show", action="store_true", help="Show matplotlib figures interactively.")

    args = p.parse_args()
    if args.input_mode == "multispectral" and not args.data_root:
        p.error("--data-root is required when --input-mode multispectral")
    if args.input_mode == "rgb" and not args.rgb_image_path:
        p.error("--rgb-image-path is required when --input-mode rgb")
    if args.seed_source == "manual_centers" and not (args.bbox_file or args.interactive_bboxes):
        p.error("--seed-source manual_centers requires --bbox-file or --interactive-bboxes")
    return args


# ---------------------------------------------------------------------------
# I/O and model helpers
# ---------------------------------------------------------------------------


def get_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        print(f"===> Using GPU {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    print("===> Using CPU")
    return torch.device("cpu")


def load_models(model_path: str, device: torch.device):
    """Load FamNet models lazily so manual-center mode can run without them."""
    from model import CountRegressor, Resnet50FPN

    weights_path = Path(model_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    resnet = Resnet50FPN().to(device).eval()
    regressor = CountRegressor(6, pool="mean").to(device).eval()
    state = torch.load(weights_path, map_location=device)
    regressor.load_state_dict(state)
    return resnet, regressor


def _load_band(root: str | Path, band: str, date: str) -> np.ndarray:
    path = Path(root) / f"{band}_{date}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing band file: {path}")
    return np.asarray(Image.open(path))


def read_data(root: str | Path, dates: Sequence[str], bands: Sequence[str]) -> np.ndarray:
    """Read multispectral data as [T, H, W, B]."""
    tensors = []
    for date in dates:
        planes = [_load_band(root, band, date) for band in bands]
        shapes = {arr.shape for arr in planes}
        if len(shapes) != 1:
            raise ValueError(f"Band shape mismatch for {date}: {sorted(shapes)}")
        tensors.append(np.stack(planes, axis=-1))
    return np.stack(tensors, axis=0)


def read_rgb_image(image_path: str | Path) -> np.ndarray:
    """Read a single RGB image as [1, H, W, 3]."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"RGB image not found: {path}")
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr[None, ...]


def crop_data(data: np.ndarray, crop: Sequence[int] | None) -> np.ndarray:
    if crop is None:
        return data
    r0, r1, c0, c1 = [int(v) for v in crop]
    if not (0 <= r0 < r1 <= data.shape[1] and 0 <= c0 < c1 <= data.shape[2]):
        raise ValueError(f"Invalid crop {crop} for data shape {data.shape}")
    return data[:, r0:r1, c0:c1, :]


def rgb_for_visualization(candidates: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([candidates["Red"], candidates["Green"], candidates["Blue"]], axis=-1)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def load_and_prepare_input(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Load input data and return both cropped raw data and normalized features.

    Vegetation indices should be computed from raw multispectral values. The
    normalized copy is used for candidate ranking, FamNet input, and clustering
    features.
    """
    if args.input_mode == "multispectral":
        raw = read_data(args.data_root, dates=args.dates, bands=args.bands)
    else:
        raw = read_rgb_image(args.rgb_image_path)
    cropped_raw = crop_data(raw, args.crop)
    normalized = normalize_per_band(cropped_raw)
    return cropped_raw, normalized


def choose_candidate_features(xn: np.ndarray, rank_day: int, prop_day: int):
    candidates_rank = build_candidate_features(xn, day=rank_day)
    candidates_prop = build_candidate_features(xn, day=prop_day)
    masks = mask_bright_spots(candidates_rank)
    ranked = rank_candidates(candidates_rank, masks, sort_by="contrast_ratio")
    if not ranked:
        raise RuntimeError("No candidate features could be ranked")
    best_names = [str(ranked[0]["name"]), str(ranked[1]["name"]), str(ranked[2]["name"]) ]
    return candidates_rank, candidates_prop, masks, ranked, best_names


def auto_exemplar_boxes_from_feature(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    args: argparse.Namespace,
) -> tuple[List[List[int]], float]:
    selected_mask, blob_info, contours = detect_big_round_blobs(
        mask,
        min_area=args.auto_min_blob_area,
        min_circularity=args.auto_min_circularity,
        min_solidity=args.auto_min_solidity,
    )
    _, top_contours = top_k_blobs(blob_info, contours, args.top_k_blobs)
    _, contours_no_border = filter_border_contours(selected_mask.shape, top_contours)
    boxes, min_box_area = get_blob_bounding_boxes(contours_no_border)
    boxes = validate_bounding_boxes(boxes, image_shape, clip=True)
    return boxes, min_box_area


def get_exemplar_boxes(
    args: argparse.Namespace,
    xn: np.ndarray,
    vis_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    best_name: str,
) -> tuple[List[List[int]], str, float]:
    """Return boxes from manual input or automatic blob detection."""
    h, w = vis_rgb.shape[:2]
    output_dir = Path(args.output_dir)
    manual_boxes: List[List[int]] = []

    if args.bbox_file is not None:
        raw_boxes = read_bbox_file(args.bbox_file, image_shape=None)
        manual_boxes = adjust_boxes_for_crop(
            raw_boxes,
            args.crop,
            (h, w),
            coordinate_space=args.bbox_coordinate_space,
        )
    elif args.interactive_bboxes:
        manual_boxes = select_or_load_exemplar_boxes(
            vis_rgb,
            interactive=True,
            output_bbox_file=output_dir / "manual_exemplar_boxes.txt",
        )

    if manual_boxes:
        min_area = min((y2 - y1 + 1) * (x2 - x1 + 1) for y1, x1, y2, x2 in manual_boxes)
        return manual_boxes, "manual", float(min_area)

    auto_boxes, min_area = auto_exemplar_boxes_from_feature(masks[best_name], (h, w), args)

    if auto_boxes:
        return auto_boxes, "automatic", float(min_area)

    if args.input_mode == "rgb":
        raise RuntimeError(
            "No automatic exemplar blobs found for RGB input. Provide --bbox-file or use --interactive-bboxes."
        )
    raise RuntimeError("No valid exemplar blobs found after filtering")


def estimate_peaks_and_radii(
    args: argparse.Namespace,
    best_feature: np.ndarray,
    exemplar_boxes: Sequence[Sequence[int]],
    box_source: str,
    vegetation_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """Estimate crown peaks and support radii using manual boxes or FamNet density."""
    h, w = best_feature.shape[:2]
    circle_debug = None

    #if args.seed_source == "manual_centers":
    #    peaks_rc = boxes_to_centers_rc(exemplar_boxes)
    #    density_map = np.zeros((h, w), dtype=np.float32)
    #    for r, c in peaks_rc:
    #        if 0 <= r < h and 0 <= c < w:
    #            density_map[r, c] = 1.0
    #    radii = box_radius_estimates(
    #        exemplar_boxes,
    #        scale=args.manual_box_radius_scale * args.neighborhood_scale,
    #        min_radius=max(3.0, args.default_radius * 0.25),
    #    )
    #    return peaks_rc, radii, density_map, circle_debug

    resnet, regressor = load_models(args.model_path, device)
    cache_name = f"density_{args.input_mode}_{box_source}_adapt{int(args.adapt)}.pt"
    density_map = run_famnet(
        best_feature,
        exemplar_boxes,
        resnet,
        regressor,
        args.load_density,
        args,
        device,
        cache_name=cache_name,
    )
    density_smooth = ndi.gaussian_filter(density_map, sigma=3)
    peaks_rc = detect_strong_peaks(
        density_smooth,
        min_distance=args.peak_min_distance,
        percentile=args.peak_percentile,
    )

    if len(peaks_rc) == 0 and args.fallback_to_manual_centers and len(exemplar_boxes) > 0:
        peaks_rc = boxes_to_centers_rc(exemplar_boxes)

    if len(peaks_rc) == 0:
        raise RuntimeError("No peaks detected in density map")

    circle_labels, circle_info, circle_debug = grow_peak_circles_until_collision(
        peaks_rc=peaks_rc,
        vegetation_mask=vegetation_mask,
        max_intersection_frac=args.max_intersection_frac,
        radius_step=args.circle_radius_step,
        initial_radius=args.circle_initial_radius,
    )
    del circle_labels
    radii = np.asarray([float(info["radius"]) for info in circle_info], dtype=np.float32)
    if len(radii) != len(peaks_rc) or np.any(radii <= 0):
        radii = np.full(len(peaks_rc), float(args.default_radius), dtype=np.float32)
    radii = np.maximum(radii * float(args.neighborhood_scale), float(args.default_radius) * 0.25)
    return peaks_rc, radii.astype(np.float32), density_map, circle_debug


def run_selected_clustering(
    method: str,
    image: np.ndarray,
    seed_clusters_rc: List[np.ndarray],
    vegetation_mask: np.ndarray,
    valid_cluster_radii: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    common = dict(
        image=image,
        seed_clusters_rc=seed_clusters_rc,
        vegetation_mask=vegetation_mask,
        neighborhood_radius=None,
        neighborhood_radii=valid_cluster_radii,
        max_seeds_per_cluster=args.max_seeds_per_cluster,
    )

    if method == "watershed":
        return propagate_labels_watershed(
            **common,
            enforce_label_neighborhoods=True,
            gradient_smoothing_sigma=args.watershed_gradient_sigma,
            distance_weight=args.watershed_distance_weight,
            compactness=args.watershed_compactness,
        )
    if method == "random_walker":
        return propagate_labels_random_walker(
            **common,
            beta=args.rw_beta,
            enforce_label_neighborhoods=True,
            crop_to_active_bbox=True,
            use_probability_constraints=True,
        )
    if method == "label_spreading":
        return propagate_labels_sklearn_label_spreading(
            **common,
            spatial_weight=0.25,
            gamma=20.0,
            alpha=0.2,
            max_iter=1000,
            enforce_neighborhoods=True,
        )
    if method == "gl_poisson":
        return propagate_labels_gl_poisson(
            **common,
            max_iter=1000,
            enforce_neighborhoods=True,
        )
    if method == "gl_laplace":
        return propagate_labels_gl_laplace(
            **common,
            enforce_neighborhoods=True,
        )
    if method == "gl_laplace_poisson_reweighted":
        return propagate_labels_gl_laplace_poisson_reweighted(
            image,
            seed_clusters_rc,
            vegetation_mask,
            max_seeds_per_cluster=args.max_seeds_per_cluster,
            neighborhood_radius=None,
            neighborhood_radii=valid_cluster_radii,
            enforce_neighborhoods=True,
        )
    raise ValueError(f"Unknown clustering method: {method}")


def save_outputs(
    args: argparse.Namespace,
    labels: np.ndarray,
    vis_rgb: np.ndarray,
    seed_points_rc: np.ndarray,
    exemplar_boxes: Sequence[Sequence[int]],
    crown_boxes: Sequence[Sequence[int]],
    best_feature: np.ndarray,
    density_map: np.ndarray,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "crown_labels.npy", labels)
    write_bbox_file(exemplar_boxes, output_dir / "exemplar_boxes_used.txt")
    write_bbox_file(crown_boxes, output_dir / "crown_boxes_from_labels.txt")
    save_label_overlay(vis_rgb, labels, seed_points_rc, output_dir / "crown_label_overlay.png")
    show_final_results(
        best_feature,
        density_map,
        vis_rgb,
        labels,
        seed_points_rc,
        save_path=output_dir / "pipeline_diagnostics.png",
        show=args.show,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = get_device(args.gpu_id)

    raw_cropped, xn = load_and_prepare_input(args)
    rank_day = int(np.clip(args.day_for_ranking, 0, xn.shape[0] - 1))
    prop_day = int(np.clip(args.day_for_propagation, 0, xn.shape[0] - 1))


    scene_image_for_mask = raw_cropped[prop_day]
    vegetation_mask, vegetation_score = compute_scene_vegetation_mask(
        scene_image=scene_image_for_mask,
        input_mode=args.input_mode,
        ground_method=args.ground_method,
        mask_mode=args.vegetation_mask_mode,
        remove_shadow=not args.disable_shadow_removal,
    )
    del vegetation_score
    vegetation_mask = vegetation_mask.astype(bool)
    if vegetation_mask.mean() < 0.001:
        raise RuntimeError("Vegetation mask is nearly empty. Try --vegetation-mask-mode all.")

    cands_rank, cands_prop, masks, ranked, best_names = choose_candidate_features(xn, rank_day, prop_day)
    print(f"Best feature for exemplar ranking: {best_names}")
    print("Top ranked features:")
    for row in ranked[:5]:
        print(
            f"  {row['name']}: contrast={float(row['contrast_ratio']):.3f}, "
            f"effect={float(row['effect_size']):.3f}, blobs={int(row['n_blobs'])}"
        )

    propagation_image = stack_feature_maps(cands_prop)
    rank_image = stack_feature_maps(cands_rank)
    h, w, c = propagation_image.shape
    print(h)
    print(w)
    print(c)

    exemplar_boxes, min_box_area = detect_closed_boundary_boxes(
        image=rank_image,
        valid_mask=vegetation_mask,
        gaussian_sigma=1.2,
        threshold_method="otsu",
        # Local closing only, not global.
        local_close_kernel_size=5,
        local_close_iterations=1,
        fill_holes=True,
        min_area=40,
        min_width=5,
        min_height=5,
        # Keep irregular crowns.
        min_circularity=0.10,
        min_solidity=0.30,
        # Remove smallest and biggest boxes.
        filter_by_mean_area=True,
        area_ratio_range=(0.6, 2.4),
        plot=True,
        plot_save_path="output/boundary_boxes_debug.png",
        return_debug=False,
    )
    box_source = "auto"

    print("Boxes:", exemplar_boxes)
    print("Min box area:", min_box_area)

    #os.exit()


    vis_rgb = rgb_for_visualization(cands_prop)
    #exemplar_boxes, box_source, min_box_area = get_exemplar_boxes(args, xn, vis_rgb, masks, best_names[0])
    #print(f"Using {len(exemplar_boxes)} {box_source} exemplar boxes")

    
    best_img_chnls = np.stack([cands_rank[best_names[0]], cands_rank[best_names[1]], cands_rank[best_names[2]]], -1)
    peaks_rc, radii, density_map, circle_debug = estimate_peaks_and_radii(
        args,
        best_img_chnls,
        exemplar_boxes,
        box_source,
        vegetation_mask,
        device,
    )
    print(f"Estimated crowns/seeds: {len(peaks_rc)}")

    clustering_image = reduce_feature_cube(
        propagation_image,
        n_components= c,
        valid_mask=vegetation_mask,
        method=args.feature_reduction,
    )
    seed_clusters_rc, valid_peaks_rc, valid_radii = sample_seed_clusters_from_peaks(
        peaks_rc=peaks_rc,
        radii=radii,
        image_shape=(h, w),
        vegetation_mask=vegetation_mask,
        num_points=args.num_exemplars,
        radius_fraction=0.3,
        random_seed=12345,
    )
    if len(seed_clusters_rc) == 0:
        raise RuntimeError("No valid seed clusters remain after vegetation masking")

    seed_points_rc = np.vstack(seed_clusters_rc) if seed_clusters_rc else np.empty((0, 2), dtype=np.int32)
    print(np.shape(clustering_image))
    labels = run_selected_clustering(
        method=args.clustering_method,
        image=clustering_image,
        seed_clusters_rc=seed_clusters_rc,
        vegetation_mask=vegetation_mask,
        valid_cluster_radii=valid_radii,
        args=args,
    )
    labels = postprocess_crown_labels(
        labels,
        min_area=args.min_label_area,
        keep_largest_component=args.keep_largest_component,
        fill_holes=args.fill_holes,
    )
    crown_boxes = labels_to_bounding_boxes(labels, min_area=args.min_label_area)
    print(f"Final crown clusters: {len(crown_boxes)}")

    save_outputs(
        args,
        labels,
        vis_rgb,
        seed_points_rc,
        exemplar_boxes,
        crown_boxes,
        minmax_scale(best_img_chnls ),
        density_map,
    )

    patch_metadata = save_clustered_tree_crown_patches(
        image_rgb=vis_rgb,
        labels=labels,
        output_dir="output/crown_patch_library",
        min_area=30,
        padding=6,
        source_image_id="source_image_001",
    )
    
    print(f"Saved {len(patch_metadata)} crown patches.")
    


    if args.save_debug and circle_debug is not None:
        plot_grown_circles_debug(
            image_rgb=vis_rgb,
            peaks_rc=valid_peaks_rc,
            labels=labels,
            circle_info=[
                {"circle_id": i + 1, "center_rc": tuple(map(int, rc)), "radius": int(r), "stop_reason": "used"}
                for i, (rc, r) in enumerate(zip(valid_peaks_rc, valid_radii))
            ],
            debug=circle_debug,
            save_path=Path(args.output_dir) / "circle_growth_debug.png",
            show=args.show,
        )

    print(f"Saved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
