from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class FitResult:
    polygon: np.ndarray
    scale_x: float
    scale_y: float
    rotation_deg: float
    score: float
    overlap_ratio: float
    precision_ratio: float
    recall_ratio: float


def polygon_centroid(polygon_xy: np.ndarray) -> np.ndarray:
    return polygon_xy.mean(axis=0)


def rasterize_polygon(
    polygon_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([tuple(point) for point in polygon_xy], outline=1, fill=1)
    return np.array(canvas, dtype=bool)


def transform_polygon(
    template_polygon: np.ndarray,
    anchor_xy: tuple[float, float],
    scale_x: float,
    scale_y: float,
    rotation_deg: float,
    anchor_mode: str = "centroid",
) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    if anchor_mode == "centroid":
        reference_point = polygon_centroid(template_polygon)
    elif anchor_mode == "apex":
        reference_point = template_polygon[np.argmin(template_polygon[:, 1])]
    else:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

    centered_template = template_polygon - reference_point
    scaled = centered_template * np.array([scale_x, scale_y], dtype=np.float64)
    rotated = scaled @ rotation.T
    return rotated + np.array(anchor_xy, dtype=np.float64)


def score_candidate(
    candidate_mask: np.ndarray,
    vegetation_mask: np.ndarray,
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

    # Practical vegetation-mask fitting objective:
    # reward the template staying inside vegetation while also preferring
    # candidates that explain a noticeable chunk of the mask.
    score = (0.60 * precision) + (0.30 * iou) + (0.10 * recall)
    return score, iou, precision, recall


def fit_pine_canopy_fasm(
    template_polygon: np.ndarray,
    vegetation_mask: np.ndarray,
    anchor_xy: tuple[float, float],
    scale_x_values: Iterable[float],
    scale_y_values: Iterable[float],
    rotation_values_deg: Iterable[float],
    anchor_mode: str = "centroid",
) -> FitResult:
    best: FitResult | None = None
    for scale_x in scale_x_values:
        for scale_y in scale_y_values:
            for rotation_deg in rotation_values_deg:
                polygon = transform_polygon(
                    template_polygon=template_polygon,
                    anchor_xy=anchor_xy,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    rotation_deg=rotation_deg,
                    anchor_mode=anchor_mode,
                )
                candidate_mask = rasterize_polygon(polygon, vegetation_mask.shape)
                score, iou, precision, recall = score_candidate(
                    candidate_mask=candidate_mask,
                    vegetation_mask=vegetation_mask,
                )
                if best is None or score > best.score:
                    best = FitResult(
                        polygon=polygon,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        rotation_deg=rotation_deg,
                        score=score,
                        overlap_ratio=iou,
                        precision_ratio=precision,
                        recall_ratio=recall,
                    )

    if best is None:
        raise ValueError("No valid candidate was produced.")
    return best


def make_general_pine_template() -> np.ndarray:
    # Positive y goes downward in image coordinates.
    return np.array(
        [
            [0.0, 0.0],
            [-10.0, 10.0],
            [-24.0, 32.0],
            [-42.0, 64.0],
            [-54.0, 98.0],
            [-42.0, 132.0],
            [-18.0, 162.0],
            [0.0, 176.0],
            [18.0, 162.0],
            [42.0, 132.0],
            [54.0, 98.0],
            [42.0, 64.0],
            [24.0, 32.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


def make_synthetic_vegetation_mask(
    image_shape: tuple[int, int],
    canopy_polygon: np.ndarray,
) -> np.ndarray:
    vegetation = rasterize_polygon(canopy_polygon, image_shape)

    # Extra vegetation blobs simulate nearby understory and neighboring crowns.
    clutter_polygons = [
        np.array([[80, 210], [120, 195], [145, 235], [112, 265], [76, 248]]),
        np.array([[250, 180], [286, 160], [314, 188], [300, 228], [258, 222]]),
        np.array([[182, 68], [196, 58], [210, 72], [206, 92], [188, 96]]),
    ]
    for poly in clutter_polygons:
        vegetation |= rasterize_polygon(poly, image_shape)

    return vegetation


def plot_demo(
    vegetation_mask: np.ndarray,
    fitted_polygon: np.ndarray,
    anchor_xy: tuple[float, float],
    output_path: Path,
) -> None:
    fitted_mask = rasterize_polygon(fitted_polygon, vegetation_mask.shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = vegetation_mask.shape
    panel_gap = 12
    panel_w = w
    panel_h = h
    title_h = 30
    canvas = Image.new(
        "RGB",
        (panel_w * 3 + panel_gap * 4, panel_h + title_h + panel_gap * 2),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    def panel_origin(index: int) -> tuple[int, int]:
        x0 = panel_gap + index * (panel_w + panel_gap)
        y0 = panel_gap + title_h
        return x0, y0

    def add_panel(image_array: np.ndarray, index: int, title: str) -> None:
        x0, y0 = panel_origin(index)
        image = Image.fromarray(image_array)
        canvas.paste(image, (x0, y0))
        draw.rectangle(
            [x0, y0, x0 + panel_w, y0 + panel_h], outline=(80, 80, 80), width=1
        )
        draw.text((x0, panel_gap), title, fill=(20, 20, 20))

    vegetation_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    vegetation_rgb[..., 1] = vegetation_mask.astype(np.uint8) * 180
    add_panel(vegetation_rgb, 0, "Vegetation Mask + Center Anchor")

    fitted_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fitted_rgb[..., 2] = fitted_mask.astype(np.uint8) * 200
    add_panel(fitted_rgb, 1, "Best Pine Template Fit")

    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_rgb[..., 1] = vegetation_mask.astype(np.uint8) * 180
    overlay_rgb[..., 2] = fitted_mask.astype(np.uint8) * 200
    add_panel(overlay_rgb, 2, "Overlay")

    for index, line_color in enumerate([(255, 0, 0), (255, 220, 0), (255, 255, 255)]):
        x0, y0 = panel_origin(index)
        shifted_polygon = fitted_polygon + np.array([x0, y0], dtype=np.float64)
        polygon_points = [tuple(point) for point in shifted_polygon]
        draw.line(polygon_points + [polygon_points[0]], fill=line_color, width=2)
        anchor_x = x0 + anchor_xy[0]
        anchor_y = y0 + anchor_xy[1]
        radius = 4
        draw.ellipse(
            [
                anchor_x - radius,
                anchor_y - radius,
                anchor_x + radius,
                anchor_y + radius,
            ],
            fill=(255, 0, 0),
            outline=(255, 255, 255),
        )

        centroid_xy = polygon_centroid(fitted_polygon)
        centroid_x = x0 + centroid_xy[0]
        centroid_y = y0 + centroid_xy[1]
        centroid_radius = 3
        draw.ellipse(
            [
                centroid_x - centroid_radius,
                centroid_y - centroid_radius,
                centroid_x + centroid_radius,
                centroid_y + centroid_radius,
            ],
            fill=(255, 255, 0),
            outline=(0, 0, 0),
        )

    canvas.save(output_path)


def run_demo() -> None:
    image_shape = (320, 320)
    anchor_xy = (170.0, 140.0)

    template_polygon = make_general_pine_template()

    synthetic_true_polygon = transform_polygon(
        template_polygon=template_polygon,
        anchor_xy=anchor_xy,
        scale_x=1.18,
        scale_y=1.04,
        rotation_deg=-6.0,
        anchor_mode="centroid",
    )
    vegetation_mask = make_synthetic_vegetation_mask(
        image_shape=image_shape,
        canopy_polygon=synthetic_true_polygon,
    )

    print(template_polygon.shape)
    print(template_polygon.shape)

    result = fit_pine_canopy_fasm(
        template_polygon=template_polygon,
        vegetation_mask=vegetation_mask,
        anchor_xy=anchor_xy,
        scale_x_values=np.linspace(0.8, 1.4, 19),
        scale_y_values=np.linspace(0.8, 1.3, 16),
        rotation_values_deg=np.linspace(-12.0, 12.0, 17),
        anchor_mode="centroid",
    )

    print("Best fit")
    print(f"  scale_x: {result.scale_x:.3f}")
    print(f"  scale_y: {result.scale_y:.3f}")
    print(f"  rotation_deg: {result.rotation_deg:.3f}")
    print(f"  score: {result.score:.4f}")
    print(f"  IoU: {result.overlap_ratio:.4f}")
    print(f"  precision_inside_vegetation: {result.precision_ratio:.4f}")
    print(f"  recall_of_total_vegetation: {result.recall_ratio:.4f}")

    output_path = Path("./workspace/output/pine_fasm_demo_result.png")
    plot_demo(
        vegetation_mask=vegetation_mask,
        fitted_polygon=result.polygon,
        anchor_xy=anchor_xy,
        output_path=output_path,
    )
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    run_demo()
