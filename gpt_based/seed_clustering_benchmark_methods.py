"""
Seed-based graph clustering / label propagation benchmark methods.

This module is designed for image-like crown segmentation experiments where you
already have sparse seed clusters and want to compare several seeded propagation
methods on the same feature image.

Primary dependency for the graph-based methods:
    pip install graphlearning

Optional baseline dependencies:
    pip install scikit-learn scikit-image

Typical use from another script
-------------------------------

    from seed_clustering_benchmark_methods import run_seed_clustering_benchmark

    results = run_seed_clustering_benchmark(
        image=feature_cube,                  # H x W x C or H x W
        seed_clusters_rc=seed_clusters_rc,   # list of arrays, each N_i x 2 as row,col
        valid_mask=vegetation_mask,          # H x W bool
        methods=[
            "gl_graph_nn",
            "gl_laplace",
            "gl_wnll",
            "gl_poisson",
            "gl_peikonal",
            "gl_poisson_mbo",
        ],
        k=20,
        spatial_weight=0.25,
        max_seeds_per_cluster=64,
    )

    labels_image = results["gl_poisson"].labels_image  # 0 background, 1..K seed labels

Seed format
-----------
seed_clusters_rc should be a sequence where each item contains pixel seed
coordinates for one cluster/crown:

    seed_clusters_rc = [
        np.array([[r0, c0], [r1, c1], ...]),  # cluster 1
        np.array([[r2, c2], [r3, c3], ...]),  # cluster 2
        ...
    ]

The output label image uses 0 for invalid/background and 1..K for clusters.
Internally, GraphLearning receives labels 0..K-1, as required by its SSL API.

Implemented methods
-------------------
GraphLearning methods:
    gl_graph_nn        Graph geodesic nearest-neighbor classifier.
    gl_laplace         Graph Laplace learning.
    gl_wnll            Weighted nonlocal Laplacian / WNLL reweighted Laplace.
    gl_laplace_poisson Poisson-reweighted Laplace learning.
    gl_poisson         Poisson learning.
    gl_peikonal        Graph p-eikonal classifier.
    gl_poisson_mbo     Poisson MBO, using class priors.

Extra non-GraphLearning baselines:
    sk_label_propagation  scikit-learn LabelPropagation.
    sk_label_spreading    scikit-learn LabelSpreading.
    skimage_random_walker scikit-image random walker for image-grid propagation.

Notes for crown experiments
---------------------------
For image segmentation, a kNN graph using only spectral/features may connect
far-away but feature-similar pixels. Set spatial_weight > 0 to append normalized
(row, col) coordinates to the feature vectors. This often improves crown-local
propagation.

If you have per-crown spatial supports/neighborhoods, pass label_supports as a
sequence of H x W boolean masks. The module will post-constrain predictions so
label j cannot occupy pixels outside support j. This is useful for benchmarking
methods fairly against a random-walker implementation with crown-local radii.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import sparse
from scipy import ndimage as ndi

try:  # Optional at import time; required only for gl_* methods.
    import graphlearning as gl  # type: ignore
except Exception:  # pragma: no cover - import availability depends on environment.
    gl = None  # type: ignore


ArrayLike = np.ndarray


@dataclass
class PreparedSeedProblem:
    """Flattened representation of an image seed-propagation problem."""

    image_shape: tuple[int, int]
    valid_mask: np.ndarray
    node_indices: np.ndarray          # flat H*W indices for valid nodes
    flat_to_node: np.ndarray          # length H*W, -1 for invalid pixels
    features: np.ndarray              # n_valid x n_features
    train_ind: np.ndarray             # node indices among valid nodes
    train_labels0: np.ndarray         # GraphLearning labels, 0..K-1
    num_classes: int
    label_supports_nodes: np.ndarray | None = None  # n_valid x K bool, optional
    graph_cache: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringResult:
    """Result returned for every benchmark method."""

    method: str
    labels_image: np.ndarray          # H x W, 0 background, 1..K cluster labels
    labels_nodes0: np.ndarray         # n_valid, 0..K-1 for valid assigned nodes, -1 unassigned
    elapsed_s: float
    info: dict[str, Any] = field(default_factory=dict)


def _as_3d_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image[..., None]
    if image.ndim == 3:
        return image
    raise ValueError(f"image must be HxW or HxWxC; got shape {image.shape}")


def _minmax_scale_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust column-wise min-max scaling to [0, 1]."""
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    finite = np.isfinite(out)
    for j in range(out.shape[1]):
        col = out[:, j]
        ok = finite[:, j]
        if not np.any(ok):
            out[:, j] = 0.0
            continue
        lo = np.min(col[ok])
        hi = np.max(col[ok])
        scale = hi - lo
        if scale <= eps:
            out[:, j] = 0.0
        else:
            out[:, j] = (col - lo) / scale
        out[~ok, j] = 0.0
    return out


def _append_spatial_features(
    features: np.ndarray,
    node_indices: np.ndarray,
    image_shape: tuple[int, int],
    spatial_weight: float,
) -> np.ndarray:
    if spatial_weight <= 0:
        return features
    h, w = image_shape
    rows, cols = np.unravel_index(node_indices, (h, w))
    denom_r = max(h - 1, 1)
    denom_c = max(w - 1, 1)
    xy = np.column_stack((rows / denom_r, cols / denom_c)).astype(np.float64)
    xy *= float(spatial_weight)
    return np.column_stack((features, xy))


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


def prepare_seed_problem(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    valid_mask: np.ndarray | None = None,
    *,
    spatial_weight: float = 0.0,
    normalize_features: bool = True,
    max_seeds_per_cluster: int | None = 64,
    label_supports: Sequence[np.ndarray] | None = None,
) -> PreparedSeedProblem:
    """Prepare image, mask, and seeds for graph-based label propagation.

    Parameters
    ----------
    image:
        H x W or H x W x C feature image.
    seed_clusters_rc:
        Sequence of arrays, one per cluster. Each array should contain row/col
        seed coordinates with shape N_i x 2.
    valid_mask:
        H x W boolean mask. Propagation is restricted to True pixels. If None,
        all finite pixels are used.
    spatial_weight:
        If > 0, append normalized row/col coordinates multiplied by this value.
    normalize_features:
        If True, min-max normalize each feature channel over valid nodes.
    max_seeds_per_cluster:
        Deterministically cap seed count per class. Set None to use all seeds.
    label_supports:
        Optional sequence of H x W boolean masks, one per class. Predictions can
        be constrained so class j only occupies support j.
    """
    img3 = _as_3d_image(image)
    h, w, c = img3.shape

    if valid_mask is None:
        valid_mask_arr = np.all(np.isfinite(img3), axis=2)
    else:
        valid_mask_arr = np.asarray(valid_mask, dtype=bool)
        if valid_mask_arr.shape != (h, w):
            raise ValueError(f"valid_mask must have shape {(h, w)}; got {valid_mask_arr.shape}")

    node_indices = np.flatnonzero(valid_mask_arr.ravel())
    if len(node_indices) == 0:
        raise ValueError("valid_mask contains no valid pixels")

    flat_to_node = np.full(h * w, -1, dtype=np.int64)
    flat_to_node[node_indices] = np.arange(len(node_indices), dtype=np.int64)

    features = img3.reshape(-1, c)[node_indices].astype(np.float64)
    features[~np.isfinite(features)] = 0.0
    if normalize_features:
        features = _minmax_scale_columns(features)
    features = _append_spatial_features(features, node_indices, (h, w), spatial_weight)

    train_nodes: list[int] = []
    train_labels: list[int] = []
    for label0, cluster_points in enumerate(seed_clusters_rc):
        points = _dedupe_and_cap_seeds(cluster_points, max_seeds_per_cluster)
        if points.size == 0:
            continue
        rr = points[:, 0]
        cc = points[:, 1]
        inside = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        if not np.any(inside):
            continue
        rr = rr[inside]
        cc = cc[inside]
        flat = rr * w + cc
        nodes = flat_to_node[flat]
        nodes = nodes[nodes >= 0]
        if len(nodes) == 0:
            continue
        nodes = np.unique(nodes)
        train_nodes.extend(nodes.tolist())
        train_labels.extend([label0] * len(nodes))

    train_ind = np.asarray(train_nodes, dtype=np.int64)
    train_labels0 = np.asarray(train_labels, dtype=np.int64)
    if len(train_ind) == 0:
        raise ValueError("No seed points fall inside valid_mask")

    num_classes = len(seed_clusters_rc)
    if num_classes <= 0:
        raise ValueError("seed_clusters_rc must contain at least one cluster")

    present_classes = np.unique(train_labels0)
    if len(present_classes) < num_classes:
        missing = sorted(set(range(num_classes)) - set(present_classes.tolist()))
        raise ValueError(
            "At least one valid seed is required for every cluster. "
            f"Missing 0-based classes: {missing}"
        )

    label_supports_nodes: np.ndarray | None = None
    if label_supports is not None:
        if len(label_supports) != num_classes:
            raise ValueError(
                f"label_supports must contain {num_classes} masks; got {len(label_supports)}"
            )
        supports = []
        for support in label_supports:
            support_arr = np.asarray(support, dtype=bool)
            if support_arr.shape != (h, w):
                raise ValueError(f"Each support mask must have shape {(h, w)}; got {support_arr.shape}")
            supports.append(support_arr.ravel()[node_indices])
        label_supports_nodes = np.column_stack(supports)
        # Make sure seed labels remain legal even if the supplied supports are a bit tight.
        label_supports_nodes[train_ind, train_labels0] = True

    return PreparedSeedProblem(
        image_shape=(h, w),
        valid_mask=valid_mask_arr,
        node_indices=node_indices,
        flat_to_node=flat_to_node,
        features=features,
        train_ind=train_ind,
        train_labels0=train_labels0,
        num_classes=num_classes,
        label_supports_nodes=label_supports_nodes,
    )


def labels_nodes_to_image(problem: PreparedSeedProblem, labels_nodes0: np.ndarray) -> np.ndarray:
    """Convert node labels 0..K-1 / -1 to H x W image labels 1..K / 0."""
    h, w = problem.image_shape
    out = np.zeros(h * w, dtype=np.int32)
    labels_nodes0 = np.asarray(labels_nodes0, dtype=np.int64)
    if labels_nodes0.shape[0] != problem.node_indices.shape[0]:
        raise ValueError("labels_nodes0 length does not match the number of valid nodes")
    assigned = labels_nodes0 >= 0
    out[problem.node_indices[assigned]] = labels_nodes0[assigned].astype(np.int32) + 1
    return out.reshape(h, w)


def _require_graphlearning() -> Any:
    if gl is None:
        raise ImportError(
            "graphlearning is required for gl_* methods. Install it with: pip install graphlearning"
        )
    return gl


def build_graphlearning_matrices(
    features: np.ndarray,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    build_distance: bool = False,
) -> tuple[Any, Any | None]:
    """Build GraphLearning kNN similarity and optional distance matrices."""
    gl_mod = _require_graphlearning()
    n = int(features.shape[0])
    if n < 2:
        raise ValueError("At least two valid nodes are required to build a graph")
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




def get_graphlearning_matrices(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    build_distance: bool = False,
) -> tuple[Any, Any | None]:
    """Cached GraphLearning kNN matrices for fairer method benchmarking.

    The first call builds the requested graph; later methods with the same
    configuration reuse it. This avoids benchmarking the identical kNN search
    repeatedly for every propagation method.
    """
    key = (int(k), str(weight_kernel), str(similarity), bool(build_distance))
    if key not in problem.graph_cache:
        problem.graph_cache[key] = build_graphlearning_matrices(
            problem.features,
            k=k,
            weight_kernel=weight_kernel,
            similarity=similarity,
            build_distance=build_distance,
        )
    return problem.graph_cache[key]

def _uniform_class_priors(num_classes: int) -> np.ndarray:
    return np.ones(num_classes, dtype=np.float64) / max(num_classes, 1)


def _class_priors_from_argument(
    class_priors: str | Sequence[float] | np.ndarray | None,
    num_classes: int,
) -> np.ndarray | None:
    if class_priors is None:
        return None
    if isinstance(class_priors, str):
        if class_priors.lower() == "uniform":
            return _uniform_class_priors(num_classes)
        raise ValueError("class_priors string must be 'uniform' or None")
    priors = np.asarray(class_priors, dtype=np.float64)
    if priors.shape != (num_classes,):
        raise ValueError(f"class_priors must have shape {(num_classes,)}; got {priors.shape}")
    s = float(np.sum(priors))
    if s <= 0:
        raise ValueError("class_priors must sum to a positive value")
    return priors / s


def _constrained_prediction_from_scores(
    scores: np.ndarray,
    supports: np.ndarray | None,
    *,
    similarity: bool = True,
) -> np.ndarray:
    """Predict labels from scores, optionally respecting per-class supports."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be n_nodes x n_classes")
    if supports is None:
        return np.argmax(scores, axis=1) if similarity else np.argmin(scores, axis=1)
    supports = np.asarray(supports, dtype=bool)
    if supports.shape != scores.shape:
        raise ValueError(f"supports shape {supports.shape} does not match scores shape {scores.shape}")
    labels = np.full(scores.shape[0], -1, dtype=np.int64)
    valid_any = np.any(supports, axis=1)
    if similarity:
        masked = np.where(supports, scores, -np.inf)
        labels[valid_any] = np.argmax(masked[valid_any], axis=1)
    else:
        masked = np.where(supports, scores, np.inf)
        labels[valid_any] = np.argmin(masked[valid_any], axis=1)
    return labels


def _fit_graphlearning_model(
    problem: PreparedSeedProblem,
    model: Any,
    *,
    enforce_supports: bool = True,
    method_name: str,
) -> ClusteringResult:
    start = time.perf_counter()
    scores = model.fit(problem.train_ind, problem.train_labels0)
    supports = problem.label_supports_nodes if enforce_supports else None
    labels0 = _constrained_prediction_from_scores(
        np.asarray(scores),
        supports,
        similarity=bool(getattr(model, "similarity", True)),
    )
    labels0[problem.train_ind] = problem.train_labels0  # Seeds are hard labels.
    labels_image = labels_nodes_to_image(problem, labels0)
    elapsed = time.perf_counter() - start
    return ClusteringResult(
        method=method_name,
        labels_image=labels_image,
        labels_nodes0=labels0,
        elapsed_s=elapsed,
        info={"graphlearning_model_name": getattr(model, "name", method_name)},
    )


def cluster_gl_graph_nearest_neighbor(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    similarity: str = "euclidean",
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Graph geodesic nearest-neighbor propagation using GraphLearning."""
    gl_mod = _require_graphlearning()
    _W, D = get_graphlearning_matrices(
        problem,
        k=k,
        weight_kernel="gaussian",
        similarity=similarity,
        build_distance=True,
    )
    model = gl_mod.ssl.graph_nearest_neighbor(D)
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_graph_nn")


def cluster_gl_laplace(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    tau: float = 0.0,
    order: int = 1,
    tol: float = 1e-5,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Graph Laplace learning from seeds."""
    gl_mod = _require_graphlearning()
    W, _D = get_graphlearning_matrices(problem, k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _class_priors_from_argument(class_priors, problem.num_classes)
    model = gl_mod.ssl.laplace(W, class_priors=priors, tau=tau, order=order, tol=tol)
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_laplace")


def cluster_gl_wnll(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    tau: float = 0.0,
    order: int = 1,
    tol: float = 1e-5,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Weighted nonlocal Laplacian via GraphLearning Laplace reweighting='wnll'."""
    gl_mod = _require_graphlearning()
    W, _D = get_graphlearning_matrices(problem, k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _class_priors_from_argument(class_priors, problem.num_classes)
    model = gl_mod.ssl.laplace(
        W,
        class_priors=priors,
        reweighting="wnll",
        tau=tau,
        order=order,
        tol=tol,
    )
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_wnll")


def cluster_gl_laplace_poisson_reweighted(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    tau: float = 0.0,
    order: int = 1,
    tol: float = 1e-5,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Laplace learning with GraphLearning's Poisson reweighting near seeds."""
    gl_mod = _require_graphlearning()
    W, _D = get_graphlearning_matrices(problem, k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _class_priors_from_argument(class_priors, problem.num_classes)
    model = gl_mod.ssl.laplace(
        W,
        class_priors=priors,
        reweighting="poisson",
        tau=tau,
        order=order,
        tol=tol,
    )
    return _fit_graphlearning_model(
        problem,
        model,
        enforce_supports=enforce_supports,
        method_name="gl_laplace_poisson",
    )


def cluster_gl_poisson(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = None,
    solver: str = "conjugate_gradient",
    p: float = 1,
    tol: float = 1e-3,
    max_iter: int = 1000,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Poisson learning on the seed graph."""
    gl_mod = _require_graphlearning()
    W, _D = get_graphlearning_matrices(problem, k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _class_priors_from_argument(class_priors, problem.num_classes)
    model = gl_mod.ssl.poisson(
        W,
        class_priors=priors,
        solver=solver,
        p=p,
        tol=tol,
        max_iter=max_iter,
    )
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_poisson")


def cluster_gl_peikonal(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    p: float = 1,
    alpha: float = 1,
    max_num_it: int = 100000,
    tol: float = 1e-3,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Graph p-eikonal classifier using GraphLearning."""
    gl_mod = _require_graphlearning()
    W, D = get_graphlearning_matrices(
        problem,
        k=k,
        weight_kernel=weight_kernel,
        similarity=similarity,
        build_distance=True,
    )
    model = gl_mod.ssl.peikonal(W, D=D, p=p, alpha=alpha, max_num_it=max_num_it, tol=tol)
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_peikonal")


def cluster_gl_poisson_mbo(
    problem: PreparedSeedProblem,
    *,
    k: int = 20,
    weight_kernel: str = "gaussian",
    similarity: str = "euclidean",
    class_priors: str | Sequence[float] | np.ndarray | None = "uniform",
    solver: str = "conjugate_gradient",
    tol: float = 1e-3,
    max_iter: int = 1000,
    Ns: int = 40,
    mu: float = 1,
    T: int = 20,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Poisson MBO seed propagation. Uses uniform priors by default."""
    gl_mod = _require_graphlearning()
    W, _D = get_graphlearning_matrices(problem, k=k, weight_kernel=weight_kernel, similarity=similarity)
    priors = _class_priors_from_argument(class_priors, problem.num_classes)
    if priors is None:
        priors = _uniform_class_priors(problem.num_classes)
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
    return _fit_graphlearning_model(problem, model, enforce_supports=enforce_supports, method_name="gl_poisson_mbo")


def _sklearn_graph_labels(problem: PreparedSeedProblem) -> np.ndarray:
    y = np.full(problem.features.shape[0], -1, dtype=np.int64)
    y[problem.train_ind] = problem.train_labels0
    return y


def cluster_sklearn_label_propagation(
    problem: PreparedSeedProblem,
    *,
    gamma: float = 20.0,
    max_iter: int = 1000,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Extra baseline: scikit-learn RBF LabelPropagation on node features."""
    try:
        from sklearn.semi_supervised import LabelPropagation
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for sk_label_propagation") from exc
    start = time.perf_counter()
    model = LabelPropagation(kernel="rbf", gamma=gamma, max_iter=max_iter)
    labels0 = model.fit(problem.features, _sklearn_graph_labels(problem)).transduction_.astype(np.int64)
    if enforce_supports and problem.label_supports_nodes is not None:
        labels0 = _constrained_prediction_from_scores(
            model.label_distributions_, problem.label_supports_nodes, similarity=True
        )
    labels0[problem.train_ind] = problem.train_labels0
    elapsed = time.perf_counter() - start
    return ClusteringResult(
        method="sk_label_propagation",
        labels_image=labels_nodes_to_image(problem, labels0),
        labels_nodes0=labels0,
        elapsed_s=elapsed,
        info={"backend": "sklearn.semi_supervised.LabelPropagation"},
    )


def cluster_sklearn_label_spreading(
    problem: PreparedSeedProblem,
    *,
    gamma: float = 20.0,
    alpha: float = 0.2,
    max_iter: int = 1000,
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Extra baseline: scikit-learn RBF LabelSpreading on node features."""
    try:
        from sklearn.semi_supervised import LabelSpreading
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for sk_label_spreading") from exc
    start = time.perf_counter()
    model = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha, max_iter=max_iter)
    labels0 = model.fit(problem.features, _sklearn_graph_labels(problem)).transduction_.astype(np.int64)
    if enforce_supports and problem.label_supports_nodes is not None:
        labels0 = _constrained_prediction_from_scores(
            model.label_distributions_, problem.label_supports_nodes, similarity=True
        )
    labels0[problem.train_ind] = problem.train_labels0
    elapsed = time.perf_counter() - start
    return ClusteringResult(
        method="sk_label_spreading",
        labels_image=labels_nodes_to_image(problem, labels0),
        labels_nodes0=labels0,
        elapsed_s=elapsed,
        info={"backend": "sklearn.semi_supervised.LabelSpreading"},
    )


def cluster_skimage_random_walker(
    problem: PreparedSeedProblem,
    *,
    image: np.ndarray | None = None,
    beta: float = 100.0,
    mode: str = "cg_mg",
    enforce_supports: bool = True,
    **_: Any,
) -> ClusteringResult:
    """Extra image-grid baseline: scikit-image random walker.

    For this baseline, pass the original H x W x C image via `image=` when
    calling the method directly. When called through `run_seed_clustering_benchmark`,
    this is supplied automatically.
    """
    if image is None:
        raise ValueError("cluster_skimage_random_walker requires the original image= argument")
    try:
        from skimage.segmentation import random_walker
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-image is required for skimage_random_walker") from exc

    start = time.perf_counter()
    img3 = _as_3d_image(image)
    h, w = problem.image_shape
    markers = np.full((h, w), -1, dtype=np.int32)
    markers[problem.valid_mask] = 0
    seed_flat = problem.node_indices[problem.train_ind]
    rr, cc = np.unravel_index(seed_flat, (h, w))
    markers[rr, cc] = problem.train_labels0.astype(np.int32) + 1

    # scikit-image expects multichannel data as H x W x C with channel_axis=-1.
    data = img3.astype(np.float64)
    try:
        labels_img = random_walker(data, markers, beta=beta, mode=mode, channel_axis=-1)
    except TypeError:
        # Older scikit-image versions used multichannel=True.
        labels_img = random_walker(data, markers, beta=beta, mode=mode, multichannel=True)
    except Exception:
        labels_img = random_walker(data, markers, beta=beta, mode="cg", channel_axis=-1)

    labels_img = labels_img.astype(np.int32)
    labels_img[~problem.valid_mask] = 0

    labels0 = np.full(problem.node_indices.shape[0], -1, dtype=np.int64)
    node_labels = labels_img.ravel()[problem.node_indices].astype(np.int64)
    assigned = node_labels > 0
    labels0[assigned] = node_labels[assigned] - 1

    if enforce_supports and problem.label_supports_nodes is not None:
        valid = labels0 >= 0
        invalid = valid & (~problem.label_supports_nodes[np.arange(len(labels0)), np.maximum(labels0, 0)])
        labels0[invalid] = -1
        labels_img = labels_nodes_to_image(problem, labels0)

    labels0[problem.train_ind] = problem.train_labels0
    labels_img = labels_nodes_to_image(problem, labels0)
    elapsed = time.perf_counter() - start
    return ClusteringResult(
        method="skimage_random_walker",
        labels_image=labels_img,
        labels_nodes0=labels0,
        elapsed_s=elapsed,
        info={"backend": "skimage.segmentation.random_walker", "beta": beta},
    )


MethodFunction = Callable[..., ClusteringResult]

METHOD_REGISTRY: dict[str, MethodFunction] = {
    "gl_graph_nn": cluster_gl_graph_nearest_neighbor,
    "gl_laplace": cluster_gl_laplace,
    "gl_wnll": cluster_gl_wnll,
    "gl_laplace_poisson": cluster_gl_laplace_poisson_reweighted,
    "gl_poisson": cluster_gl_poisson,
    "gl_peikonal": cluster_gl_peikonal,
    "gl_poisson_mbo": cluster_gl_poisson_mbo,
    "sk_label_propagation": cluster_sklearn_label_propagation,
    "sk_label_spreading": cluster_sklearn_label_spreading,
    "skimage_random_walker": cluster_skimage_random_walker,
}

DEFAULT_GL_METHODS = (
    "gl_graph_nn",
    "gl_laplace",
    "gl_wnll",
    "gl_poisson",
    "gl_peikonal",
    "gl_poisson_mbo",
)


def run_seed_clustering_benchmark(
    image: np.ndarray,
    seed_clusters_rc: Sequence[np.ndarray],
    valid_mask: np.ndarray | None = None,
    *,
    methods: Sequence[str] = DEFAULT_GL_METHODS,
    label_supports: Sequence[np.ndarray] | None = None,
    spatial_weight: float = 0.0,
    normalize_features: bool = True,
    max_seeds_per_cluster: int | None = 64,
    k: int = 20,
    method_params: Mapping[str, Mapping[str, Any]] | None = None,
    continue_on_error: bool = True,
    enforce_supports: bool = True,
    **shared_params: Any,
) -> dict[str, ClusteringResult]:
    """Run several seeded clustering methods on the same prepared problem.

    Parameters
    ----------
    image, seed_clusters_rc, valid_mask:
        See `prepare_seed_problem`.
    methods:
        Names from METHOD_REGISTRY.
    label_supports:
        Optional per-class legal support masks.
    spatial_weight:
        Weight for appended spatial coordinates. For crown segmentation, values
        in roughly [0.05, 1.0] are useful to benchmark.
    method_params:
        Dict mapping method name to method-specific keyword arguments.
    continue_on_error:
        If True, store an error result and continue. If False, raise.
    enforce_supports:
        Whether to enforce label_supports when provided.
    shared_params:
        Extra keyword arguments passed to every method. Method-specific params
        override shared params.

    Returns
    -------
    dict[str, ClusteringResult]
        Result per method. Failed methods have info["error"] set and an all-zero
        label image.
    """
    problem = prepare_seed_problem(
        image,
        seed_clusters_rc,
        valid_mask,
        spatial_weight=spatial_weight,
        normalize_features=normalize_features,
        max_seeds_per_cluster=max_seeds_per_cluster,
        label_supports=label_supports,
    )
    method_params = method_params or {}
    results: dict[str, ClusteringResult] = {}

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            raise KeyError(f"Unknown method '{method_name}'. Available: {sorted(METHOD_REGISTRY)}")
        params: dict[str, Any] = {"k": k, "enforce_supports": enforce_supports, **shared_params}
        params.update(method_params.get(method_name, {}))
        if method_name == "skimage_random_walker":
            params.setdefault("image", image)
        start = time.perf_counter()
        try:
            results[method_name] = METHOD_REGISTRY[method_name](problem, **params)
        except Exception as exc:
            if not continue_on_error:
                raise
            elapsed = time.perf_counter() - start
            h, w = problem.image_shape
            results[method_name] = ClusteringResult(
                method=method_name,
                labels_image=np.zeros((h, w), dtype=np.int32),
                labels_nodes0=np.full(problem.node_indices.shape[0], -1, dtype=np.int64),
                elapsed_s=elapsed,
                info={"error": repr(exc)},
            )
    return results


def result_summary(results: Mapping[str, ClusteringResult]) -> list[dict[str, Any]]:
    """Compact serializable summary for printing/logging."""
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        assigned = int(np.sum(result.labels_image > 0))
        unique = sorted(np.unique(result.labels_image[result.labels_image > 0]).tolist())
        row = {
            "method": name,
            "elapsed_s": result.elapsed_s,
            "assigned_pixels": assigned,
            "labels_present": unique,
        }
        if "error" in result.info:
            row["error"] = result.info["error"]
        rows.append(row)
    return rows


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
    base = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    for points, rad in zip(seed_clusters_rc, radii):
        mask = np.zeros((h, w), dtype=bool)
        r_float = float(rad)
        for r, c in _dedupe_and_cap_seeds(np.asarray(points), None):
            if 0 <= r < h and 0 <= c < w:
                mask |= (yy - int(r)) ** 2 + (xx - int(c)) ** 2 <= r_float ** 2
        supports.append(mask & base)
    return supports


def demo_synthetic() -> dict[str, ClusteringResult]:
    """Small synthetic demo that runs optional non-GL baselines everywhere.

    GraphLearning methods are included only if graphlearning is installed. This
    keeps the file executable in minimal environments.
    """
    h, w = 64, 64
    yy, xx = np.mgrid[:h, :w]
    image = np.zeros((h, w, 2), dtype=np.float64)
    image[..., 0] = np.exp(-((yy - 20) ** 2 + (xx - 22) ** 2) / 180.0)
    image[..., 1] = np.exp(-((yy - 42) ** 2 + (xx - 43) ** 2) / 220.0)
    valid_mask = image.sum(axis=2) > 0.05
    seeds = [np.array([[20, 22], [21, 23], [19, 21]]), np.array([[42, 43], [43, 44], [41, 42]])]
    supports = make_disk_supports((h, w), seeds, radius=22, valid_mask=valid_mask)

    methods = ["sk_label_propagation", "sk_label_spreading", "skimage_random_walker"]
    if gl is not None:
        methods = list(DEFAULT_GL_METHODS) + methods

    return run_seed_clustering_benchmark(
        image,
        seeds,
        valid_mask,
        methods=methods,
        label_supports=supports,
        spatial_weight=0.25,
        k=12,
        continue_on_error=True,
    )


def main() -> None:
    """Run the synthetic demo and print a compact benchmark summary."""
    results = demo_synthetic()
    for row in result_summary(results):
        print(row)


if __name__ == "__main__":
    main()
