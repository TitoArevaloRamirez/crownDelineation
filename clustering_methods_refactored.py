"""Semi-supervised crown label propagation methods.

All methods use a common seed-cluster interface:
    seed_clusters_rc = [array([[row, col], ...]), ...]
The output is an int32 label image where 0 is background and 1..K are crown ids.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import random_walker, watershed

from utils_refactored import build_cluster_neighborhood_masks, normalize_channels
import graphlearning as gl  # type: ignore


# ---------------------------------------------------------------------------
# Shared validation and preparation helpers
# ---------------------------------------------------------------------------


def _as_3d_image(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        return x
    raise ValueError(f"image must have shape HxW or HxWxC, got {x.shape}")


def _check_mask_shape(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {m.shape}")
    return m


def _dedupe_and_cap_seeds(points_rc: np.ndarray, max_seeds: int | None) -> np.ndarray:
    pts = np.asarray(points_rc)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pts = np.asarray(pts, dtype=np.int64).reshape(-1, 2)
    pts = np.unique(pts, axis=0)
    if max_seeds is not None and len(pts) > int(max_seeds):
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        pts = pts[order]
        keep = np.linspace(0, len(pts) - 1, int(max_seeds)).round().astype(int)
        pts = pts[keep]
    return pts


def _valid_seed_points(
    cluster: np.ndarray,
    allowed: np.ndarray,
    *,
    max_seeds: int | None,
) -> np.ndarray:
    h, w = allowed.shape
    pts = _dedupe_and_cap_seeds(cluster, max_seeds)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.int32)
    rr, cc = pts[:, 0], pts[:, 1]
    inside = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
    pts = pts[inside]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.int32)
    pts = pts[allowed[pts[:, 0], pts[:, 1]]]
    return pts.astype(np.int32, copy=False)


def _build_markers_and_supports(
    image_shape: tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    neighborhood_radius: int | float | None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None,
    max_seeds_per_cluster: int | None,
    restrict_to_neighborhood_union: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create markers, label supports, active mask, and positive label ids."""
    h, w = image_shape
    veg = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")
    n_clusters = len(seed_clusters_rc)
    supports = build_cluster_neighborhood_masks(
        image_shape=(h, w),
        seed_clusters_rc=seed_clusters_rc,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
    ).astype(bool)
    if supports.shape != (n_clusters, h, w):
        raise ValueError(f"Expected supports shape {(n_clusters, h, w)}, got {supports.shape}")
    supports &= veg[None, :, :]

    allowed = veg & np.any(supports, axis=0) if restrict_to_neighborhood_union else veg.copy()
    markers = np.zeros((h, w), dtype=np.int32)
    markers[~allowed] = -1

    positive_labels: list[int] = []
    for label_id, cluster in enumerate(seed_clusters_rc, start=1):
        pts = _valid_seed_points(
            cluster,
            supports[label_id - 1],
            max_seeds=max_seeds_per_cluster,
        )
        if len(pts) == 0:
            continue
        markers[pts[:, 0], pts[:, 1]] = label_id
        positive_labels.append(label_id)

    if positive_labels:
        structure = ndi.generate_binary_structure(2, 1)
        cc, _ = ndi.label(allowed, structure=structure)
        seeded_cc = np.unique(cc[markers > 0])
        seeded_cc = seeded_cc[seeded_cc > 0]
        if len(seeded_cc):
            reachable = np.isin(cc, seeded_cc)
            markers[allowed & ~reachable] = -1
            allowed &= reachable
        else:
            allowed[:] = False
            markers[:, :] = -1

    return markers, supports, allowed, np.asarray(positive_labels, dtype=np.int32)


def _crop_to_active(
    arrays: Sequence[np.ndarray],
    active_mask: np.ndarray,
) -> tuple[list[np.ndarray], tuple[slice, slice]]:
    rows, cols = np.where(active_mask)
    if len(rows) == 0:
        row_slice = slice(0, active_mask.shape[0])
        col_slice = slice(0, active_mask.shape[1])
    else:
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    cropped: list[np.ndarray] = []
    for arr in arrays:
        if arr.ndim == 2:
            cropped.append(arr[row_slice, col_slice])
        elif arr.ndim == 3 and arr.shape[0] == active_mask.shape[0]:
            cropped.append(arr[row_slice, col_slice, :])
        elif arr.ndim == 3:
            cropped.append(arr[:, row_slice, col_slice])
        else:
            raise ValueError(f"Unsupported crop array shape: {arr.shape}")
    return cropped, (row_slice, col_slice)


def make_label_supports(
    image_shape: tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build K x H x W boolean support masks for label-specific constraints."""
    h, w = image_shape
    veg = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")
    supports = build_cluster_neighborhood_masks(
        image_shape=(h, w),
        seed_clusters_rc=seed_clusters_rc,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
    ).astype(bool)
    supports &= veg[None, :, :]
    return supports


# ---------------------------------------------------------------------------
# Random walker and watershed methods
# ---------------------------------------------------------------------------


def propagate_labels_random_walker(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    beta: float = 100.0,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    max_seeds_per_cluster: int | None = 64,
    enforce_label_neighborhoods: bool = True,
    crop_to_active_bbox: bool = True,
    use_probability_constraints: bool = True,
) -> np.ndarray:
    """Conservative random-walker crown propagation."""
    img3 = _as_3d_image(image).astype(np.float32)
    h, w, _ = img3.shape
    if len(seed_clusters_rc) == 0:
        return np.zeros((h, w), dtype=np.int32)

    markers, supports, allowed, positive_labels = _build_markers_and_supports(
        (h, w),
        seed_clusters_rc,
        vegetation_mask,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        max_seeds_per_cluster=max_seeds_per_cluster,
        restrict_to_neighborhood_union=True,
    )
    if len(positive_labels) == 0 or not np.any(allowed):
        return np.zeros((h, w), dtype=np.int32)

    data = normalize_channels(img3)
    data[~np.isfinite(data)] = 0.0

    if crop_to_active_bbox:
        (data_crop, markers_crop, supports_crop, veg_crop), slices = _crop_to_active(
            [data, markers, supports, np.asarray(vegetation_mask, dtype=bool)],
            allowed | (markers > 0),
        )
        row_slice, col_slice = slices
    else:
        data_crop = data
        markers_crop = markers
        supports_crop = supports
        veg_crop = np.asarray(vegetation_mask, dtype=bool)
        row_slice, col_slice = slice(0, h), slice(0, w)

    labels_present = np.unique(markers_crop[markers_crop > 0]).astype(np.int32)
    if len(labels_present) == 0:
        return np.zeros((h, w), dtype=np.int32)

    def _run_random_walker(return_full_prob: bool):
        last_error: Exception | None = None
        for mode in ("cg_mg", "cg", "bf"):
            try:
                return random_walker(
                    data_crop,
                    markers_crop,
                    beta=float(beta),
                    mode=mode,
                    channel_axis=-1,
                    copy=True,
                    return_full_prob=return_full_prob,
                )
            except TypeError:
                try:
                    return random_walker(
                        data_crop,
                        markers_crop,
                        beta=float(beta),
                        mode=mode,
                        multichannel=True,
                        return_full_prob=return_full_prob,
                    )
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
        raise RuntimeError("random_walker failed for all modes") from last_error

    want_probabilities = bool(enforce_label_neighborhoods and use_probability_constraints)
    if want_probabilities:
        try:
            probabilities = np.asarray(_run_random_walker(return_full_prob=True), dtype=np.float32)
            if probabilities.shape[0] != len(labels_present):
                raise RuntimeError(
                    f"Unexpected probability shape {probabilities.shape}; expected {len(labels_present)} planes"
                )
            label_supports = supports_crop[labels_present - 1]
            label_allowed = label_supports & veg_crop[None, :, :]
            constrained = probabilities.copy()
            constrained[~label_allowed] = -np.inf
            has_candidate = np.any(label_allowed, axis=0) & (markers_crop != -1)
            labels_crop = np.zeros(markers_crop.shape, dtype=np.int32)
            if np.any(has_candidate):
                best_idx = np.argmax(constrained[:, has_candidate], axis=0)
                labels_crop[has_candidate] = labels_present[best_idx]
        except Exception as exc:
            print(
                "Warning: probability-constrained random walker failed; "
                f"falling back to hard post-filtering. Details: {exc}"
            )
            labels_crop = np.asarray(_run_random_walker(return_full_prob=False), dtype=np.int32)
    else:
        labels_crop = np.asarray(_run_random_walker(return_full_prob=False), dtype=np.int32)

    labels_crop[markers_crop == -1] = 0
    labels_crop[~veg_crop] = 0
    if enforce_label_neighborhoods:
        for label_id in labels_present:
            labels_crop[(labels_crop == label_id) & (~supports_crop[label_id - 1])] = 0
    labels_crop[markers_crop > 0] = markers_crop[markers_crop > 0]

    labels = np.zeros((h, w), dtype=np.int32)
    labels[row_slice, col_slice] = labels_crop
    return labels


def propagate_labels_random_walker_plain(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    beta: float = 100.0,
    mode_sequence: Sequence[str] = ("cg_mg", "cg", "bf"),
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    """Plain random walker with optional label support cleanup."""
    img3 = _as_3d_image(image).astype(np.float32)
    h, w, _ = img3.shape
    veg = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")
    markers = np.zeros((h, w), dtype=np.int32)
    markers[~veg] = -1

    for label_id, cluster in enumerate(seed_clusters_rc, start=1):
        pts = _valid_seed_points(cluster, veg, max_seeds=max_seeds_per_cluster)
        if len(pts):
            markers[pts[:, 0], pts[:, 1]] = label_id
    if not np.any(markers > 0):
        return np.zeros((h, w), dtype=np.int32)

    data = normalize_channels(img3)
    data[~np.isfinite(data)] = 0.0
    last_error: Exception | None = None
    labels: np.ndarray | None = None
    for mode in mode_sequence:
        try:
            labels = random_walker(data, markers, beta=float(beta), mode=mode, channel_axis=-1, copy=True).astype(np.int32)
            break
        except TypeError:
            try:
                labels = random_walker(data, markers, beta=float(beta), mode=mode, multichannel=True).astype(np.int32)
                break
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc
    if labels is None:
        raise RuntimeError("random_walker failed for all requested modes") from last_error

    labels[~veg] = 0
    if enforce_neighborhoods:
        supports = make_label_supports((h, w), seed_clusters_rc, veg, neighborhood_radius, neighborhood_radii)
        for label_id in range(1, len(seed_clusters_rc) + 1):
            labels[(labels == label_id) & (~supports[label_id - 1])] = 0
    return labels.astype(np.int32, copy=False)


def propagate_labels_watershed(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    max_seeds_per_cluster: int | None = 64,
    enforce_label_neighborhoods: bool = True,
    crop_to_active_bbox: bool = True,
    gradient_smoothing_sigma: float = 1.0,
    distance_weight: float = 0.15,
    compactness: float = 0.0,
    watershed_line: bool = False,
) -> np.ndarray:
    """Marker-controlled watershed constrained by vegetation and label supports."""
    img3 = _as_3d_image(image).astype(np.float32)
    h, w, _ = img3.shape
    if len(seed_clusters_rc) == 0:
        return np.zeros((h, w), dtype=np.int32)

    markers, supports, allowed, labels_present = _build_markers_and_supports(
        (h, w),
        seed_clusters_rc,
        vegetation_mask,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        max_seeds_per_cluster=max_seeds_per_cluster,
        restrict_to_neighborhood_union=True,
    )
    if len(labels_present) == 0 or not np.any(allowed):
        return np.zeros((h, w), dtype=np.int32)

    data = normalize_channels(img3)
    data[~np.isfinite(data)] = 0.0
    if gradient_smoothing_sigma is not None and float(gradient_smoothing_sigma) > 0:
        sigma = (float(gradient_smoothing_sigma), float(gradient_smoothing_sigma), 0.0)
        data_grad = ndi.gaussian_filter(data, sigma=sigma)
    else:
        data_grad = data

    elevation = np.zeros((h, w), dtype=np.float32)
    for ch in range(data_grad.shape[-1]):
        gr, gc = np.gradient(data_grad[:, :, ch])
        elevation += gr.astype(np.float32) ** 2 + gc.astype(np.float32) ** 2
    elevation = np.sqrt(elevation).astype(np.float32, copy=False)
    finite = np.isfinite(elevation)
    if np.any(finite):
        lo, hi = float(elevation[finite].min()), float(elevation[finite].max())
        elevation = (elevation - lo) / (hi - lo) if hi > lo else np.zeros_like(elevation)
    elevation[~np.isfinite(elevation)] = 0.0

    if distance_weight is not None and float(distance_weight) > 0:
        distance = ndi.distance_transform_edt(allowed).astype(np.float32)
        max_dist = float(distance.max())
        if max_dist > 0:
            elevation = elevation - float(distance_weight) * (distance / max_dist)

    if crop_to_active_bbox:
        (elev_crop, markers_crop, allowed_crop, supports_crop, veg_crop), slices = _crop_to_active(
            [elevation, markers, allowed, supports, np.asarray(vegetation_mask, dtype=bool)],
            allowed | (markers > 0),
        )
        row_slice, col_slice = slices
    else:
        elev_crop = elevation
        markers_crop = markers
        allowed_crop = allowed
        supports_crop = supports
        veg_crop = np.asarray(vegetation_mask, dtype=bool)
        row_slice, col_slice = slice(0, h), slice(0, w)

    positive = np.unique(markers_crop[markers_crop > 0]).astype(np.int32)
    if len(positive) == 0:
        return np.zeros((h, w), dtype=np.int32)

    labels_crop = watershed(
        elev_crop,
        markers=markers_crop,
        mask=allowed_crop,
        compactness=float(compactness),
        watershed_line=bool(watershed_line),
    ).astype(np.int32, copy=False)
    labels_crop[~allowed_crop] = 0
    labels_crop[~veg_crop] = 0
    if enforce_label_neighborhoods:
        for label_id in positive:
            labels_crop[(labels_crop == label_id) & (~supports_crop[label_id - 1])] = 0
    labels_crop[markers_crop > 0] = markers_crop[markers_crop > 0]

    labels = np.zeros((h, w), dtype=np.int32)
    labels[row_slice, col_slice] = labels_crop
    return labels


# ---------------------------------------------------------------------------
# GraphLearning and scikit-learn propagation methods
# ---------------------------------------------------------------------------


def _minmax_scale_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            out[:, j] = 0.0
            continue
        lo, hi = float(np.min(col[finite])), float(np.max(col[finite]))
        if hi - lo <= eps:
            out[:, j] = 0.0
        else:
            out[:, j] = (col - lo) / (hi - lo)
        out[~finite, j] = 0.0
    return out


def _prepare_flat_problem(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    spatial_weight: float = 0.0,
    normalize_features: bool = True,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> dict[str, Any]:
    img3 = _as_3d_image(image).astype(np.float32)
    h, w, c = img3.shape
    veg = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")
    node_flat = np.flatnonzero(veg.ravel())
    if len(node_flat) == 0:
        raise ValueError("vegetation_mask contains no valid pixels")

    flat_to_node = np.full(h * w, -1, dtype=np.int64)
    flat_to_node[node_flat] = np.arange(len(node_flat), dtype=np.int64)

    features = img3.reshape(-1, c)[node_flat].astype(np.float64)
    features[~np.isfinite(features)] = 0.0
    if normalize_features:
        features = _minmax_scale_columns(features)
    if spatial_weight > 0:
        rows, cols = np.unravel_index(node_flat, (h, w))
        spatial = np.column_stack((rows / max(h - 1, 1), cols / max(w - 1, 1)))
        features = np.column_stack((features, float(spatial_weight) * spatial))

    train_label_by_node = np.full(len(node_flat), -1, dtype=np.int64)
    for label0, cluster in enumerate(seed_clusters_rc):
        pts = _valid_seed_points(cluster, veg, max_seeds=max_seeds_per_cluster)
        if len(pts) == 0:
            continue
        flat = pts[:, 0] * w + pts[:, 1]
        nodes = np.unique(flat_to_node[flat])
        nodes = nodes[nodes >= 0]
        train_label_by_node[nodes] = label0

    train_ind = np.flatnonzero(train_label_by_node >= 0).astype(np.int64)
    train_labels0 = train_label_by_node[train_ind].astype(np.int64)
    if len(train_ind) == 0:
        raise ValueError("No seed points fall inside vegetation_mask")

    n_clusters = len(seed_clusters_rc)
    missing = sorted(set(range(n_clusters)) - set(train_labels0.tolist()))
    if missing:
        raise ValueError(f"Each cluster needs at least one valid seed. Missing labels: {missing}")

    label_supports_nodes = None
    if enforce_neighborhoods:
        supports = make_label_supports((h, w), seed_clusters_rc, veg, neighborhood_radius, neighborhood_radii)
        label_supports_nodes = supports.reshape(n_clusters, -1)[:, node_flat].T
        label_supports_nodes[train_ind, train_labels0] = True

    return {
        "image_shape": (h, w),
        "node_flat": node_flat,
        "features": features,
        "train_ind": train_ind,
        "train_labels0": train_labels0,
        "n_clusters": n_clusters,
        "label_supports_nodes": label_supports_nodes,
    }


def _labels_nodes_to_image(problem: dict[str, Any], labels_nodes0: np.ndarray) -> np.ndarray:
    h, w = problem["image_shape"]
    node_flat = problem["node_flat"]
    labels0 = np.asarray(labels_nodes0, dtype=np.int64)
    if len(labels0) != len(node_flat):
        raise ValueError("labels_nodes0 length does not match number of valid nodes")
    out = np.zeros(h * w, dtype=np.int32)
    assigned = labels0 >= 0
    out[node_flat[assigned]] = labels0[assigned].astype(np.int32) + 1
    return out.reshape(h, w)


def _scores_to_labels(
    scores: np.ndarray,
    *,
    supports: np.ndarray | None = None,
    similarity: bool = True,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape n_nodes x n_classes, got {scores.shape}")
    if supports is None:
        return np.argmax(scores, axis=1) if similarity else np.argmin(scores, axis=1)
    supports = np.asarray(supports, dtype=bool)
    if supports.shape != scores.shape:
        raise ValueError(f"supports shape {supports.shape} does not match scores {scores.shape}")
    labels = np.full(scores.shape[0], -1, dtype=np.int64)
    has_candidate = np.any(supports, axis=1)
    if similarity:
        masked = np.where(supports, scores, -np.inf)
        labels[has_candidate] = np.argmax(masked[has_candidate], axis=1)
    else:
        masked = np.where(supports, scores, np.inf)
        labels[has_candidate] = np.argmin(masked[has_candidate], axis=1)
    return labels


def _require_graphlearning() -> Any:
    if gl is None:
        raise ImportError("graphlearning is required. Install with: pip install graphlearning")
    return gl


def _build_gl_graph(
    features: np.ndarray,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    build_distance: bool = False,
) -> tuple[Any, Any | None]:
    gl_mod = _require_graphlearning()
    n = int(features.shape[0])
    if n < 2:
        raise ValueError("At least two valid pixels are required to build a graph")
    k_eff = int(max(1, min(k, n - 1)))
    W = gl_mod.weightmatrix.knn(features, k_eff, kernel=weight_kernel, similarity=similarity, symmetrize=True)
    D = None
    if build_distance:
        D = gl_mod.weightmatrix.knn(features, k_eff, kernel="distance", similarity=similarity, symmetrize=True)
    return W, D


def _uniform_class_priors(n_clusters: int) -> np.ndarray:
    return np.ones(n_clusters, dtype=np.float64) / max(n_clusters, 1)


def _coerce_class_priors(class_priors: str | Sequence[float] | np.ndarray | None, n_clusters: int) -> np.ndarray | None:
    if class_priors is None:
        return None
    if isinstance(class_priors, str):
        if class_priors.lower() == "uniform":
            return _uniform_class_priors(n_clusters)
        raise ValueError("class_priors must be None, 'uniform', or a numeric vector")
    priors = np.asarray(class_priors, dtype=np.float64).reshape(-1)
    if len(priors) != n_clusters:
        raise ValueError(f"class_priors must have length {n_clusters}, got {len(priors)}")
    total = float(np.sum(priors))
    if total <= 0:
        raise ValueError("class_priors must sum to a positive value")
    return priors / total


def _fit_gl_model_to_image(problem: dict[str, Any], model: Any) -> np.ndarray:
    scores = model.fit(problem["train_ind"], problem["train_labels0"])
    labels0 = _scores_to_labels(
        scores,
        supports=problem["label_supports_nodes"],
        similarity=bool(getattr(model, "similarity", True)),
    )
    labels0[problem["train_ind"]] = problem["train_labels0"]
    return _labels_nodes_to_image(problem, labels0)


def propagate_labels_gl_graph_nearest_neighbor(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    k: int = 20,
    spatial_weight: float = 0.25,
    similarity: str = "euclidean",
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
    alpha: float = 1.0,
) -> np.ndarray:
    gl_mod = _require_graphlearning()
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    W, D = _build_gl_graph(problem["features"], k=k, similarity=similarity, build_distance=True)
    model = gl_mod.ssl.graph_nearest_neighbor(W, D=D, alpha=alpha)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_laplace(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    k: int = 20,
    spatial_weight: float = 0.25,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    tau: float = 0.0,
    order: int = 1,
    tol: float = 1e-5,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    gl_mod = _require_graphlearning()
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    W, _ = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(W, class_priors=priors, tau=tau, order=order, tol=tol)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_wnll(*args, **kwargs) -> np.ndarray:
    """Weighted nonlocal Laplacian via GraphLearning Laplace reweighting='wnll'."""
    gl_mod = _require_graphlearning()
    image, seed_clusters_rc, vegetation_mask = args[:3]
    kwargs_local = dict(kwargs)
    k = kwargs_local.pop("k", 20)
    spatial_weight = kwargs_local.pop("spatial_weight", 0.25)
    weight_kernel = kwargs_local.pop("weight_kernel", "gaussian")
    similarity = kwargs_local.pop("similarity", "euclidean")
    class_priors = kwargs_local.pop("class_priors", None)
    tau = kwargs_local.pop("tau", 0.0)
    order = kwargs_local.pop("order", 1)
    tol = kwargs_local.pop("tol", 1e-5)
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=kwargs_local.pop("max_seeds_per_cluster", 64),
        neighborhood_radius=kwargs_local.pop("neighborhood_radius", None),
        neighborhood_radii=kwargs_local.pop("neighborhood_radii", None),
        enforce_neighborhoods=kwargs_local.pop("enforce_neighborhoods", False),
    )
    W, _ = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(W, class_priors=priors, reweighting="wnll", tau=tau, order=order, tol=tol)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_laplace_poisson_reweighted(*args, **kwargs) -> np.ndarray:
    """Laplace learning with GraphLearning reweighting='poisson'."""
    gl_mod = _require_graphlearning()
    image, seed_clusters_rc, vegetation_mask = args[:3]
    kwargs_local = dict(kwargs)
    k = kwargs_local.pop("k", 20)
    spatial_weight = kwargs_local.pop("spatial_weight", 0.25)
    weight_kernel = kwargs_local.pop("weight_kernel", "gaussian")
    similarity = kwargs_local.pop("similarity", "euclidean")
    class_priors = kwargs_local.pop("class_priors", None)
    tau = kwargs_local.pop("tau", 0.0)
    order = kwargs_local.pop("order", 1)
    tol = kwargs_local.pop("tol", 1e-5)
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=kwargs_local.pop("max_seeds_per_cluster", 64),
        neighborhood_radius=kwargs_local.pop("neighborhood_radius", None),
        neighborhood_radii=kwargs_local.pop("neighborhood_radii", None),
        enforce_neighborhoods=kwargs_local.pop("enforce_neighborhoods", False),
    )
    W, _ = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(W, class_priors=priors, reweighting="poisson", tau=tau, order=order, tol=tol)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_poisson(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    k: int = 20,
    spatial_weight: float = 0.25,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    solver: str = "conjugate_gradient",
    p: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 1000,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    gl_mod = _require_graphlearning()
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    W, _ = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.poisson(W, class_priors=priors, solver=solver, p=p, tol=tol, max_iter=max_iter)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_peikonal(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    k: int = 20,
    spatial_weight: float = 0.25,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    p: float = 1.0,
    alpha: float = 1.0,
    max_num_it: int = 100000,
    tol: float = 1e-3,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    gl_mod = _require_graphlearning()
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    W, D = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity, build_distance=True)
    model = gl_mod.ssl.peikonal(W, D=D, p=p, alpha=alpha, max_num_it=max_num_it, tol=tol)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_poisson_mbo(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    k: int = 20,
    spatial_weight: float = 0.25,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = "uniform",
    solver: str = "conjugate_gradient",
    tol: float = 1e-3,
    max_iter: int = 1000,
    Ns: int = 40,
    mu: float = 1.0,
    T: int = 20,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    gl_mod = _require_graphlearning()
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    W, _ = _build_gl_graph(problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    if priors is None:
        priors = _uniform_class_priors(problem["n_clusters"])
    model = gl_mod.ssl.poisson_mbo(W, class_priors=priors, solver=solver, tol=tol, max_iter=max_iter, Ns=Ns, mu=mu, T=T)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_sklearn_label_propagation(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    spatial_weight: float = 0.25,
    gamma: float = 20.0,
    max_iter: int = 1000,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    try:
        from sklearn.semi_supervised import LabelPropagation
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for this method") from exc
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    y = np.full(problem["features"].shape[0], -1, dtype=np.int64)
    y[problem["train_ind"]] = problem["train_labels0"]
    model = LabelPropagation(kernel="rbf", gamma=gamma, max_iter=max_iter)
    model.fit(problem["features"], y)
    if enforce_neighborhoods and problem["label_supports_nodes"] is not None:
        labels0 = _scores_to_labels(model.label_distributions_, supports=problem["label_supports_nodes"])
    else:
        labels0 = model.transduction_.astype(np.int64)
    labels0[problem["train_ind"]] = problem["train_labels0"]
    return _labels_nodes_to_image(problem, labels0)


def propagate_labels_sklearn_label_spreading(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    spatial_weight: float = 0.25,
    gamma: float = 20.0,
    alpha: float = 0.2,
    max_iter: int = 1000,
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    try:
        from sklearn.semi_supervised import LabelSpreading
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for this method") from exc
    problem = _prepare_flat_problem(
        image,
        seed_clusters_rc,
        vegetation_mask,
        spatial_weight=spatial_weight,
        max_seeds_per_cluster=max_seeds_per_cluster,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
        enforce_neighborhoods=enforce_neighborhoods,
    )
    y = np.full(problem["features"].shape[0], -1, dtype=np.int64)
    y[problem["train_ind"]] = problem["train_labels0"]
    model = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha, max_iter=max_iter)
    model.fit(problem["features"], y)
    if enforce_neighborhoods and problem["label_supports_nodes"] is not None:
        labels0 = _scores_to_labels(model.label_distributions_, supports=problem["label_supports_nodes"])
    else:
        labels0 = model.transduction_.astype(np.int64)
    labels0[problem["train_ind"]] = problem["train_labels0"]
    return _labels_nodes_to_image(problem, labels0)
