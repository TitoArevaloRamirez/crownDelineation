from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class FASMModel:
    mean_shape: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    n_landmarks: int


@dataclass
class FitResult:
    polygon: np.ndarray
    scale: float
    rotation_deg: float
    coeffs: np.ndarray
    score: float
    iou: float
    precision: float
    recall: float


def polygon_centroid(polygon_xy: np.ndarray) -> np.ndarray:
    return polygon_xy.mean(axis=0)


def resample_closed_polygon(polygon_xy: np.ndarray, n_landmarks: int) -> np.ndarray:
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        raise ValueError("polygon_xy must have shape [N, 2].")
    if len(polygon_xy) < 3:
        raise ValueError("A polygon needs at least three vertices.")

    closed = np.vstack([polygon_xy, polygon_xy[0]])
    edges = np.diff(closed, axis=0)
    seg_lengths = np.sqrt((edges**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    perimeter = cumulative[-1]
    if perimeter <= 0:
        raise ValueError("Polygon perimeter must be positive.")

    samples = np.linspace(0.0, perimeter, n_landmarks + 1)[:-1]
    result = np.zeros((n_landmarks, 2), dtype=np.float64)

    seg_index = 0
    for i, s in enumerate(samples):
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < s:
            seg_index += 1
        start = closed[seg_index]
        end = closed[seg_index + 1]
        seg_start = cumulative[seg_index]
        seg_len = max(seg_lengths[seg_index], 1e-12)
        t = (s - seg_start) / seg_len
        result[i] = start + t * (end - start)

    return result


def normalize_shape(polygon_xy: np.ndarray) -> np.ndarray:
    centered = polygon_xy - polygon_centroid(polygon_xy)
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if rms <= 0:
        raise ValueError("Shape RMS size must be positive.")
    return centered / rms


def best_rotation_align(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    h = source_xy.T @ target_xy
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return source_xy @ r


def build_fasm_model(
    training_polygons: Sequence[np.ndarray],
    n_landmarks: int = 64,
    n_modes: int | None = None,
    variance_keep: float = 0.97,
    procrustes_iters: int = 8,
) -> FASMModel:
    if len(training_polygons) < 2:
        raise ValueError("True deformable FASM needs at least two training polygons.")

    shapes = [normalize_shape(resample_closed_polygon(poly, n_landmarks)) for poly in training_polygons]
    mean_shape = shapes[0].copy()

    for _ in range(procrustes_iters):
        aligned = [best_rotation_align(shape, mean_shape) for shape in shapes]
        mean_shape = np.mean(np.stack(aligned, axis=0), axis=0)
        mean_shape = normalize_shape(mean_shape)
        shapes = aligned

    x = np.stack([shape.reshape(-1) for shape in shapes], axis=0)
    mean_vec = mean_shape.reshape(-1)
    centered = x - mean_vec

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    if len(eigenvalues) == 0:
        raise ValueError("Training polygons produced no usable deformation modes.")

    if n_modes is None:
        explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        n_modes = int(np.searchsorted(explained, variance_keep) + 1)
    n_modes = max(1, min(n_modes, len(eigenvalues)))

    modes = eigenvectors[:, :n_modes].T.reshape(n_modes, n_landmarks, 2)
    return FASMModel(
        mean_shape=mean_shape,
        modes=modes,
        eigenvalues=eigenvalues[:n_modes],
        n_landmarks=n_landmarks,
    )


def deform_shape(model: FASMModel, coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape != (len(model.eigenvalues),):
        raise ValueError("coeffs must have shape [n_modes].")
    shape = model.mean_shape.copy()
    for i, b in enumerate(coeffs):
        shape += b * model.modes[i]
    return shape


def apply_pose(
    centered_shape_xy: np.ndarray,
    anchor_xy: tuple[float, float],
    scale: float,
    rotation_deg: float,
) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    posed = (centered_shape_xy * scale) @ rotation.T
    return posed + np.array(anchor_xy, dtype=np.float64)


def rasterize_polygon(polygon_xy: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([tuple(point) for point in polygon_xy], outline=1, fill=1)
    return np.array(canvas, dtype=bool)


def score_candidate(
    candidate_mask: np.ndarray,
    vegetation_mask: np.ndarray,
    coeffs: np.ndarray,
    eigenvalues: np.ndarray,
    shape_penalty_weight: float = 0.03,
) -> tuple[float, float, float, float]:
    candidate_area = candidate_mask.sum()
    vegetation_area = vegetation_mask.sum()
    if candidate_area == 0 or vegetation_area == 0:
        return -np.inf, 0.0, 0.0, 0.0

    intersection = np.logical_and(candidate_mask, vegetation_mask).sum()
    union = np.logical_or(candidate_mask, vegetation_mask).sum()
    precision = intersection / candidate_area
    recall = intersection / vegetation_area
    iou = intersection / union if union else 0.0

    normalized = coeffs / np.sqrt(np.maximum(eigenvalues, 1e-12))
    shape_penalty = np.sum(normalized**2)

    score = (0.60 * precision) + (0.30 * iou) + (0.10 * recall) - (shape_penalty_weight * shape_penalty)
    return score, iou, precision, recall


def clip_coeffs(coeffs: np.ndarray, eigenvalues: np.ndarray, sigma_limit: float = 3.0) -> np.ndarray:
    limits = sigma_limit * np.sqrt(np.maximum(eigenvalues, 0.0))
    return np.clip(coeffs, -limits, limits)


def evaluate_fit(
    model: FASMModel,
    vegetation_mask: np.ndarray,
    anchor_xy: tuple[float, float],
    scale: float,
    rotation_deg: float,
    coeffs: np.ndarray,
    shape_penalty_weight: float = 0.03,
) -> tuple[float, float, float, float, np.ndarray]:
    centered_shape = deform_shape(model, coeffs)
    polygon = apply_pose(centered_shape, anchor_xy=anchor_xy, scale=scale, rotation_deg=rotation_deg)
    candidate_mask = rasterize_polygon(polygon, vegetation_mask.shape)
    score, iou, precision, recall = score_candidate(
        candidate_mask=candidate_mask,
        vegetation_mask=vegetation_mask,
        coeffs=coeffs,
        eigenvalues=model.eigenvalues,
        shape_penalty_weight=shape_penalty_weight,
    )
    return score, iou, precision, recall, polygon


def fit_deformable_fasm(
    model: FASMModel,
    vegetation_mask: np.ndarray,
    anchor_xy: tuple[float, float],
    scale_values: Iterable[float],
    rotation_values_deg: Iterable[float],
    coeff_sigma_schedule: Sequence[float] = (2.0, 1.0, 0.5),
    coeff_samples: int = 7,
    shape_penalty_weight: float = 0.03,
) -> FitResult:
    scale_values = np.asarray(list(scale_values), dtype=np.float64)
    rotation_values_deg = np.asarray(list(rotation_values_deg), dtype=np.float64)

    coeffs = np.zeros(len(model.eigenvalues), dtype=np.float64)
    best_score = -np.inf
    best_scale = float(scale_values[len(scale_values) // 2])
    best_rotation = 0.0
    best_polygon = apply_pose(model.mean_shape, anchor_xy=anchor_xy, scale=best_scale, rotation_deg=best_rotation)
    best_iou = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for scale in scale_values:
        for rotation_deg in rotation_values_deg:
            score, iou, precision, recall, polygon = evaluate_fit(
                model=model,
                vegetation_mask=vegetation_mask,
                anchor_xy=anchor_xy,
                scale=float(scale),
                rotation_deg=float(rotation_deg),
                coeffs=coeffs,
                shape_penalty_weight=shape_penalty_weight,
            )
            if score > best_score:
                best_score = score
                best_scale = float(scale)
                best_rotation = float(rotation_deg)
                best_polygon = polygon
                best_iou = iou
                best_precision = precision
                best_recall = recall

    for sigma_span in coeff_sigma_schedule:
        for mode_idx, eigenvalue in enumerate(model.eigenvalues):
            sigma = np.sqrt(max(eigenvalue, 1e-12))
            trial_values = np.linspace(-sigma_span * sigma, sigma_span * sigma, coeff_samples)
            mode_best_coeff = coeffs[mode_idx]

            for trial in trial_values:
                trial_coeffs = coeffs.copy()
                trial_coeffs[mode_idx] = trial
                trial_coeffs = clip_coeffs(trial_coeffs, model.eigenvalues)

                score, iou, precision, recall, polygon = evaluate_fit(
                    model=model,
                    vegetation_mask=vegetation_mask,
                    anchor_xy=anchor_xy,
                    scale=best_scale,
                    rotation_deg=best_rotation,
                    coeffs=trial_coeffs,
                    shape_penalty_weight=shape_penalty_weight,
                )
                if score > best_score:
                    best_score = score
                    mode_best_coeff = trial_coeffs[mode_idx]
                    coeffs = trial_coeffs
                    best_polygon = polygon
                    best_iou = iou
                    best_precision = precision
                    best_recall = recall

            coeffs[mode_idx] = mode_best_coeff

        local_scales = np.linspace(best_scale * 0.92, best_scale * 1.08, 9)
        local_rotations = np.linspace(best_rotation - 6.0, best_rotation + 6.0, 13)
        for scale in local_scales:
            for rotation_deg in local_rotations:
                score, iou, precision, recall, polygon = evaluate_fit(
                    model=model,
                    vegetation_mask=vegetation_mask,
                    anchor_xy=anchor_xy,
                    scale=float(scale),
                    rotation_deg=float(rotation_deg),
                    coeffs=coeffs,
                    shape_penalty_weight=shape_penalty_weight,
                )
                if score > best_score:
                    best_score = score
                    best_scale = float(scale)
                    best_rotation = float(rotation_deg)
                    best_polygon = polygon
                    best_iou = iou
                    best_precision = precision
                    best_recall = recall

    return FitResult(
        polygon=best_polygon,
        scale=best_scale,
        rotation_deg=best_rotation,
        coeffs=coeffs.copy(),
        score=best_score,
        iou=best_iou,
        precision=best_precision,
        recall=best_recall,
    )


def fit_polygon_from_training_shapes(
    training_polygons: Sequence[np.ndarray],
    vegetation_mask: np.ndarray,
    anchor_xy: tuple[float, float],
    n_landmarks: int = 64,
    n_modes: int | None = None,
    scale_values: Iterable[float] = np.linspace(20.0, 140.0, 25),
    rotation_values_deg: Iterable[float] = np.linspace(-25.0, 25.0, 31),
) -> tuple[FASMModel, FitResult]:
    model = build_fasm_model(
        training_polygons=training_polygons,
        n_landmarks=n_landmarks,
        n_modes=n_modes,
    )
    result = fit_deformable_fasm(
        model=model,
        vegetation_mask=vegetation_mask,
        anchor_xy=anchor_xy,
        scale_values=scale_values,
        rotation_values_deg=rotation_values_deg,
    )
    return model, result


if __name__ == "__main__":
    # Minimal usage note:
    # 1. Provide multiple training polygons of pine crowns with consistent outline order.
    # 2. Build the FASM model from those polygons.
    # 3. Fit the model to a vegetation mask using the crown-center anchor.
    print("Import this module and call build_fasm_model(...) then fit_deformable_fasm(...).")
