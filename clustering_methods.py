import numpy as np
from typing import Any, List, Sequence, Tuple
from scipy import ndimage as ndi

from skimage.segmentation import random_walker, watershed

from utils import (
    normalize_channels,
    build_cluster_neighborhood_masks,
)

try:
    import graphlearning as gl  # type: ignore
except Exception:  # graphlearning is optional unless a gl_* method is called.
    gl = None  # type: ignore


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

    def _prepare_cluster_seeds(
        cluster: np.ndarray, cluster_allowed: np.ndarray
    ) -> np.ndarray:
        """Return valid, unique, optionally downsampled seed points for one crown."""
        pts = np.asarray(cluster, dtype=np.int32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = pts.reshape(-1, 2)

        in_bounds = (
            (pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)
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
            idx = np.linspace(
                0, len(pts) - 1, int(max_seeds_per_cluster), dtype=np.int32
            )
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
        raise RuntimeError(
            "random_walker failed in both cg_mg and cg modes"
        ) from last_error

    if want_probabilities:
        try:
            probabilities = np.asarray(
                _run_random_walker(return_full_prob=True), dtype=np.float32
            )

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
                best_idx = np.argmax(
                    constrained_probabilities[:, has_candidate], axis=0
                )
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


# -----------------------------------------------------------------------------
# Shared small helpers
# -----------------------------------------------------------------------------


def _as_3d_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image[..., None]
    if image.ndim == 3:
        return image
    raise ValueError(f"image must have shape HxW or HxWxC, got {image.shape}")


def _check_mask_shape(
    mask: np.ndarray, shape: tuple[int, int], name: str
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask


def _minmax_scale_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            out[:, j] = 0.0
            continue
        lo = np.min(col[finite])
        hi = np.max(col[finite])
        scale = hi - lo
        if scale <= eps:
            out[:, j] = 0.0
        else:
            out[:, j] = (col - lo) / scale
        out[~finite, j] = 0.0
    return out


def _dedupe_and_cap_seeds(points_rc: np.ndarray, max_seeds: int | None) -> np.ndarray:
    points_rc = np.asarray(points_rc)
    if points_rc.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    points_rc = np.asarray(points_rc, dtype=np.int64).reshape(-1, 2)
    points_rc = np.unique(points_rc, axis=0)

    if max_seeds is not None and len(points_rc) > int(max_seeds):
        order = np.lexsort((points_rc[:, 1], points_rc[:, 0]))
        points_rc = points_rc[order]
        keep = np.linspace(0, len(points_rc) - 1, int(max_seeds)).round().astype(int)
        points_rc = points_rc[keep]

    return points_rc


def _coerce_cluster_radii(
    radius: int | float | Sequence[int | float] | np.ndarray | None,
    n_clusters: int,
) -> np.ndarray:
    """Return one scalar radius per cluster."""
    if radius is None:
        raise ValueError("radius cannot be None here")

    if np.isscalar(radius):
        return np.full(n_clusters, float(radius), dtype=float)

    radii = np.asarray(radius, dtype=float).reshape(-1)
    if len(radii) == 1:
        return np.full(n_clusters, float(radii[0]), dtype=float)
    if len(radii) != n_clusters:
        raise ValueError(f"Expected 1 or {n_clusters} radii, got {len(radii)}")
    return radii


def make_label_supports(
    image_shape: tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build K x H x W boolean support masks for label-specific constraints.

    If neither radius argument is supplied, every label is allowed everywhere
    inside vegetation_mask.
    """
    h, w = image_shape
    vegetation_mask = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")
    n_clusters = len(seed_clusters_rc)

    if neighborhood_radii is None and neighborhood_radius is None:
        return np.broadcast_to(vegetation_mask, (n_clusters, h, w)).copy()

    supports = build_cluster_neighborhood_masks(
        image_shape=(h, w),
        seed_clusters_rc=seed_clusters_rc,
        neighborhood_radius=neighborhood_radius,
        neighborhood_radii=neighborhood_radii,
    ).astype(bool)

    if supports.shape != (n_clusters, h, w):
        raise ValueError(
            f"Expected supports shape {(n_clusters, h, w)}, got {supports.shape}"
        )

    supports &= vegetation_mask[None, :, :]
    return supports


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
    """Flatten an image seed problem for graph-based methods."""
    img3 = _as_3d_image(image)
    h, w, c = img3.shape
    vegetation_mask = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")

    node_flat = np.flatnonzero(vegetation_mask.ravel())
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
        spatial = np.column_stack(
            (
                rows / max(h - 1, 1),
                cols / max(w - 1, 1),
            )
        )
        features = np.column_stack((features, float(spatial_weight) * spatial))

    train_label_by_node = np.full(len(node_flat), -1, dtype=np.int64)

    for label0, cluster in enumerate(seed_clusters_rc):
        points = _dedupe_and_cap_seeds(cluster, max_seeds_per_cluster)
        if len(points) == 0:
            continue

        rr = points[:, 0]
        cc = points[:, 1]
        inside = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        rr = rr[inside]
        cc = cc[inside]
        if len(rr) == 0:
            continue

        flat = rr * w + cc
        nodes = flat_to_node[flat]
        nodes = np.unique(nodes[nodes >= 0])
        train_label_by_node[nodes] = label0

    train_ind = np.flatnonzero(train_label_by_node >= 0).astype(np.int64)
    train_labels0 = train_label_by_node[train_ind].astype(np.int64)

    if len(train_ind) == 0:
        raise ValueError("No seed points fall inside vegetation_mask")

    n_clusters = len(seed_clusters_rc)
    present = set(train_labels0.tolist())
    missing = sorted(set(range(n_clusters)) - present)
    if missing:
        raise ValueError(
            f"Each cluster needs at least one valid seed. Missing labels: {missing}"
        )

    label_supports_nodes = None
    if enforce_neighborhoods:
        supports = make_label_supports(
            image_shape=(h, w),
            seed_clusters_rc=seed_clusters_rc,
            vegetation_mask=vegetation_mask,
            neighborhood_radius=neighborhood_radius,
            neighborhood_radii=neighborhood_radii,
        )
        label_supports_nodes = supports.reshape(n_clusters, -1)[:, node_flat].T
        label_supports_nodes[train_ind, train_labels0] = True

    return {
        "image_shape": (h, w),
        "node_flat": node_flat,
        "features": features,
        "train_ind": train_ind,
        "train_labels0": train_labels0,
        "n_clusters": n_clusters,
        "vegetation_mask": vegetation_mask,
        "label_supports_nodes": label_supports_nodes,
    }


def _labels_nodes_to_image(
    problem: dict[str, Any], labels_nodes0: np.ndarray
) -> np.ndarray:
    h, w = problem["image_shape"]
    node_flat = problem["node_flat"]
    labels_nodes0 = np.asarray(labels_nodes0, dtype=np.int64)

    if len(labels_nodes0) != len(node_flat):
        raise ValueError("labels_nodes0 length does not match number of valid nodes")

    out = np.zeros(h * w, dtype=np.int32)
    assigned = labels_nodes0 >= 0
    out[node_flat[assigned]] = labels_nodes0[assigned].astype(np.int32) + 1
    return out.reshape(h, w)


def _require_graphlearning() -> Any:
    if gl is None:
        raise ImportError(
            "graphlearning is required. Install with: pip install graphlearning"
        )
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
    W = gl_mod.weightmatrix.knn(
        features,
        k_eff,
        kernel=weight_kernel,
        similarity=similarity,
        symmetrize=True,
    )

    D = None
    if build_distance:
        D = gl_mod.weightmatrix.knn(
            features,
            k_eff,
            kernel="distance",
            similarity=similarity,
            symmetrize=True,
        )

    return W, D


def _uniform_class_priors(n_clusters: int) -> np.ndarray:
    return np.ones(n_clusters, dtype=np.float64) / max(n_clusters, 1)


def _coerce_class_priors(
    class_priors: str | Sequence[float] | np.ndarray | None,
    n_clusters: int,
) -> np.ndarray | None:
    if class_priors is None:
        return None
    if isinstance(class_priors, str):
        if class_priors.lower() == "uniform":
            return _uniform_class_priors(n_clusters)
        raise ValueError("class_priors must be None, 'uniform', or a numeric vector")

    priors = np.asarray(class_priors, dtype=np.float64).reshape(-1)
    if len(priors) != n_clusters:
        raise ValueError(
            f"class_priors must have length {n_clusters}, got {len(priors)}"
        )
    total = float(np.sum(priors))
    if total <= 0:
        raise ValueError("class_priors must sum to a positive value")
    return priors / total


def _scores_to_labels(
    scores: np.ndarray,
    *,
    supports: np.ndarray | None = None,
    similarity: bool = True,
) -> np.ndarray:
    """Convert GraphLearning scores to 0-based node labels."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(
            f"Expected scores shape n_nodes x n_classes, got {scores.shape}"
        )

    if supports is None:
        return np.argmax(scores, axis=1) if similarity else np.argmin(scores, axis=1)

    supports = np.asarray(supports, dtype=bool)
    if supports.shape != scores.shape:
        raise ValueError(
            f"supports shape {supports.shape} does not match scores shape {scores.shape}"
        )

    labels = np.full(scores.shape[0], -1, dtype=np.int64)
    has_candidate = np.any(supports, axis=1)

    if similarity:
        masked_scores = np.where(supports, scores, -np.inf)
        labels[has_candidate] = np.argmax(masked_scores[has_candidate], axis=1)
    else:
        masked_scores = np.where(supports, scores, np.inf)
        labels[has_candidate] = np.argmin(masked_scores[has_candidate], axis=1)

    return labels


def _fit_gl_model_to_image(
    problem: dict[str, Any],
    model: Any,
) -> np.ndarray:
    """Fit a GraphLearning model and return H x W labels."""
    scores = model.fit(problem["train_ind"], problem["train_labels0"])
    labels0 = _scores_to_labels(
        scores,
        supports=problem["label_supports_nodes"],
        similarity=bool(getattr(model, "similarity", True)),
    )

    # Seeds are hard labels.
    labels0[problem["train_ind"]] = problem["train_labels0"]
    return _labels_nodes_to_image(problem, labels0)


# -----------------------------------------------------------------------------
# GraphLearning methods
# -----------------------------------------------------------------------------


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
    """Graph geodesic nearest-neighbor propagation using GraphLearning."""
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
    W, D = _build_gl_graph(
        problem["features"], k=k, similarity=similarity, build_distance=True
    )
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
    """Graph Laplace learning from seed labels."""
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
    W, _ = _build_gl_graph(
        problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity
    )
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(W, class_priors=priors, tau=tau, order=order, tol=tol)
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_wnll(
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
    """Weighted nonlocal Laplacian via GraphLearning Laplace reweighting='wnll'."""
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
    W, _ = _build_gl_graph(
        problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity
    )
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(
        W,
        class_priors=priors,
        reweighting="wnll",
        tau=tau,
        order=order,
        tol=tol,
    )
    return _fit_gl_model_to_image(problem, model)


def propagate_labels_gl_laplace_poisson_reweighted(
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
    """Laplace learning with GraphLearning reweighting='poisson'."""
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
    W, _ = _build_gl_graph(
        problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity
    )
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.laplace(
        W,
        class_priors=priors,
        reweighting="poisson",
        tau=tau,
        order=order,
        tol=tol,
    )
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
    """Poisson learning from seed labels."""
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
    W, _ = _build_gl_graph(
        problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity
    )
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    model = gl_mod.ssl.poisson(
        W,
        class_priors=priors,
        solver=solver,
        p=p,
        tol=tol,
        max_iter=max_iter,
    )
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
    """Graph p-eikonal seed classifier."""
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
    W, D = _build_gl_graph(
        problem["features"],
        k=k,
        weight_kernel=weight_kernel,
        similarity=similarity,
        build_distance=True,
    )
    model = gl_mod.ssl.peikonal(
        W, D=D, p=p, alpha=alpha, max_num_it=max_num_it, tol=tol
    )
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
    """Poisson MBO seed propagation. Uses uniform class priors by default."""
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
    W, _ = _build_gl_graph(
        problem["features"], k=k, weight_kernel=weight_kernel, similarity=similarity
    )
    priors = _coerce_class_priors(class_priors, problem["n_clusters"])
    if priors is None:
        priors = _uniform_class_priors(problem["n_clusters"])
    model = gl_mod.ssl.poisson_mbo(
        W,
        class_priors=priors,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        Ns=Ns,
        mu=mu,
        T=T,
    )
    return _fit_gl_model_to_image(problem, model)


# -----------------------------------------------------------------------------
# Scikit-learn baselines
# -----------------------------------------------------------------------------


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
    """Scikit-learn RBF LabelPropagation baseline."""
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
        labels0 = _scores_to_labels(
            model.label_distributions_, supports=problem["label_supports_nodes"]
        )
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
    """Scikit-learn RBF LabelSpreading baseline."""
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
        labels0 = _scores_to_labels(
            model.label_distributions_, supports=problem["label_supports_nodes"]
        )
    else:
        labels0 = model.transduction_.astype(np.int64)

    labels0[problem["train_ind"]] = problem["train_labels0"]
    return _labels_nodes_to_image(problem, labels0)


# -----------------------------------------------------------------------------
# Random-walker baselines
# -----------------------------------------------------------------------------


def propagate_labels_random_walker_plain(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    *,
    beta: float = 100.0,
    mode_sequence: Sequence[str] = ("cg_mg", "cg"),
    max_seeds_per_cluster: int | None = 64,
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
    enforce_neighborhoods: bool = False,
) -> np.ndarray:
    """Plain scikit-image random walker with optional post-hoc support cleanup."""
    img3 = _as_3d_image(image).astype(np.float32)
    h, w, _ = img3.shape
    vegetation_mask = _check_mask_shape(vegetation_mask, (h, w), "vegetation_mask")

    markers = np.zeros((h, w), dtype=np.int32)
    markers[~vegetation_mask] = -1

    for label_id, cluster in enumerate(seed_clusters_rc, start=1):
        pts = _dedupe_and_cap_seeds(cluster, max_seeds_per_cluster)
        if len(pts) == 0:
            continue
        rr = pts[:, 0]
        cc = pts[:, 1]
        inside = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w) & vegetation_mask[rr, cc]
        markers[rr[inside], cc[inside]] = label_id

    if not np.any(markers > 0):
        return np.zeros((h, w), dtype=np.int32)

    data = normalize_channels(img3)
    data[~np.isfinite(data)] = 0.0

    last_error: Exception | None = None
    for mode in mode_sequence:
        try:
            labels = random_walker(
                data,
                markers,
                beta=beta,
                mode=mode,
                channel_axis=-1,
                copy=True,
            ).astype(np.int32)
            break
        except TypeError:
            try:
                labels = random_walker(
                    data,
                    markers,
                    beta=beta,
                    mode=mode,
                    multichannel=True,
                ).astype(np.int32)
                break
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            "random_walker failed for all requested modes"
        ) from last_error

    labels[~vegetation_mask] = 0

    if enforce_neighborhoods:
        supports = make_label_supports(
            image_shape=(h, w),
            seed_clusters_rc=seed_clusters_rc,
            vegetation_mask=vegetation_mask,
            neighborhood_radius=neighborhood_radius,
            neighborhood_radii=neighborhood_radii,
        )
        for label_id in range(1, len(seed_clusters_rc) + 1):
            labels[(labels == label_id) & (~supports[label_id - 1])] = 0

    return labels.astype(np.int32, copy=False)


def propagate_labels_watershed(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    neighborhood_radius: int | None = None,
    neighborhood_radii: Sequence[int | float] | None = None,
    max_seeds_per_cluster: int | None = 64,
    enforce_label_neighborhoods: bool = True,
    crop_to_active_bbox: bool = True,
    gradient_smoothing_sigma: float = 1.0,
    distance_weight: float = 0.15,
    compactness: float = 0.0,
    watershed_line: bool = False,
) -> np.ndarray:
    """
    Marker-controlled watershed propagation constrained by vegetation and crown-local masks.

    This function follows the same seed-cluster interface as
    propagate_labels_random_walker(...), but uses watershed instead of random walker.

    The propagation is conservative:
      1. Pixels outside the vegetation mask are ignored.
      2. Pixels outside every crown neighborhood are ignored.
      3. Active connected components without seeds are removed.
      4. Optionally, each label is forced to remain inside its own neighborhood mask.
      5. Optionally, the solve is cropped to the active bounding box for speed.

    Watershed needs an elevation image. Here, the elevation is built from the
    gradient magnitude of the normalized feature image. High-gradient pixels act
    as barriers. A small negative distance-transform term can be added so that
    watershed basins prefer the interior of valid vegetation regions.

    Args:
        image:
            Array [H, W, C] or [H, W]. Multispectral or feature image.

        seed_clusters_rc:
            List of K arrays. Each array contains seed points for one crown,
            in (row, col) format.

        vegetation_mask:
            Boolean or uint8 array [H, W]. True/1 means valid vegetation.

        neighborhood_radius:
            Optional global radius for all clusters.

        neighborhood_radii:
            Optional independent radius per cluster.

        max_seeds_per_cluster:
            Optional cap on seed markers per crown. Set to None to use all seeds.

        enforce_label_neighborhoods:
            If True, each output label is allowed only inside its own crown-local
            neighborhood mask.

        crop_to_active_bbox:
            If True, run watershed only on the bounding box containing allowed
            pixels and seeds.

        gradient_smoothing_sigma:
            Gaussian smoothing applied to each normalized feature channel before
            computing gradients. Use 0.0 to disable smoothing.

        distance_weight:
            Weight of the normalized distance-transform term. Larger values make
            labels prefer the interior of valid regions. Use 0.0 to disable it.

        compactness:
            Passed to skimage.segmentation.watershed. Larger values make regions
            more spatially compact, but may ignore image boundaries.

        watershed_line:
            Passed to skimage.segmentation.watershed. If True, watershed boundary
            pixels are set to 0.

    Returns:
        labels:
            Integer array [H, W].
            0 = background / non-vegetation / forbidden / watershed boundary.
            1..K = propagated crown labels.
    """
    image = np.asarray(image, dtype=np.float32)
    vegetation_mask = np.asarray(vegetation_mask).astype(bool)

    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [H, W, C] or [H, W], got {image.shape}")

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

    allowed = vegetation_mask & np.any(neighborhoods, axis=0)
    if not np.any(allowed):
        print("Warning: no allowed propagation pixels after applying masks.")
        return np.zeros((h, w), dtype=np.int32)

    markers = np.zeros((h, w), dtype=np.int32)

    def _prepare_cluster_seeds(
        cluster: np.ndarray, cluster_allowed: np.ndarray
    ) -> np.ndarray:
        """Return valid, unique, optionally downsampled seed points for one crown."""
        pts = np.asarray(cluster, dtype=np.int32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = pts.reshape(-1, 2)

        in_bounds = (
            (pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)
        )
        pts = pts[in_bounds]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        pts = pts[cluster_allowed[pts[:, 0], pts[:, 1]]]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        pts = np.unique(pts, axis=0)

        if max_seeds_per_cluster is not None and len(pts) > max_seeds_per_cluster:
            order = np.lexsort((pts[:, 1], pts[:, 0]))
            pts = pts[order]
            idx = np.linspace(
                0, len(pts) - 1, int(max_seeds_per_cluster), dtype=np.int32
            )
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

    # Remove allowed connected components that do not contain any seed.
    structure = ndi.generate_binary_structure(2, 1)
    component_labels, _ = ndi.label(allowed, structure=structure)
    seeded_components = np.unique(component_labels[markers > 0])
    seeded_components = seeded_components[seeded_components > 0]

    if len(seeded_components) == 0:
        print("Warning: no seeded connected components in allowed mask.")
        return np.zeros((h, w), dtype=np.int32)

    allowed = allowed & np.isin(component_labels, seeded_components)

    data = normalize_channels(image)
    data[~np.isfinite(data)] = 0.0

    if gradient_smoothing_sigma is not None and gradient_smoothing_sigma > 0:
        sigma = (float(gradient_smoothing_sigma), float(gradient_smoothing_sigma), 0.0)
        data_for_gradient = ndi.gaussian_filter(data, sigma=sigma)
    else:
        data_for_gradient = data

    # Build a scalar watershed elevation from the multi-channel gradient magnitude.
    elevation = np.zeros((h, w), dtype=np.float32)
    for ch in range(data_for_gradient.shape[2]):
        grad_r, grad_c = np.gradient(data_for_gradient[:, :, ch])
        elevation += grad_r.astype(np.float32) ** 2 + grad_c.astype(np.float32) ** 2
    elevation = np.sqrt(elevation).astype(np.float32, copy=False)

    finite = np.isfinite(elevation)
    if np.any(finite):
        lo = float(np.min(elevation[finite]))
        hi = float(np.max(elevation[finite]))
        if hi > lo:
            elevation = (elevation - lo) / (hi - lo)
        else:
            elevation[:] = 0.0
    elevation[~np.isfinite(elevation)] = 0.0

    if distance_weight is not None and distance_weight > 0:
        distance = ndi.distance_transform_edt(allowed).astype(np.float32)
        max_distance = float(np.max(distance))
        if max_distance > 0:
            distance /= max_distance
            elevation = elevation - float(distance_weight) * distance

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

    elevation_crop = elevation[row_slice, col_slice]
    markers_crop = markers[row_slice, col_slice]
    allowed_crop = allowed[row_slice, col_slice]
    neighborhoods_crop = neighborhoods[:, row_slice, col_slice]
    vegetation_crop = vegetation_mask[row_slice, col_slice]

    positive_labels = np.unique(markers_crop[markers_crop > 0]).astype(np.int32)
    if len(positive_labels) == 0:
        print("Warning: no valid seed points after active-domain cropping.")
        return np.zeros((h, w), dtype=np.int32)

    labels_crop = watershed(
        elevation_crop,
        markers=markers_crop,
        mask=allowed_crop,
        compactness=float(compactness),
        watershed_line=bool(watershed_line),
    ).astype(np.int32, copy=False)

    # Universal cleanup.
    labels_crop[~allowed_crop] = 0
    labels_crop[~vegetation_crop] = 0

    # Hard safety net: no label is allowed outside its own crown neighborhood.
    if enforce_label_neighborhoods:
        for label_id in positive_labels:
            invalid = (labels_crop == label_id) & (~neighborhoods_crop[label_id - 1])
            labels_crop[invalid] = 0

    # Preserve seed labels exactly when they are still inside the valid crop.
    seed_pixels = markers_crop > 0
    labels_crop[seed_pixels] = markers_crop[seed_pixels]

    labels = np.zeros((h, w), dtype=np.int32)
    labels[row_slice, col_slice] = labels_crop
    return labels
