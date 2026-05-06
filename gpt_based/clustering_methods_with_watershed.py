import numpy as np
from typing import Dict, List, Sequence, Tuple
from scipy import ndimage as ndi

from skimage.segmentation import random_walker, watershed

from utils import (
    normalize_channels,
    build_cluster_neighborhood_masks,
)


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
