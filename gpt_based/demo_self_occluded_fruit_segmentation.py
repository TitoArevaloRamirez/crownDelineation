"""Minimal runnable demo for the self-occluded fruit contour segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from self_occluded_fruit_segmentation import (
    segment_self_occluded_fruit_contour,
    synthetic_overlapping_circles_mask,
)


def render_demo(output_path: str | Path = "/workspace/output/self_occluded_demo.png") -> Path:
    mask = synthetic_overlapping_circles_mask()
    result = segment_self_occluded_fruit_contour(mask)

    canvas = Image.new("RGB", (mask.shape[1] * 2 + 48, mask.shape[0] + 32), (250, 248, 243))
    draw = ImageDraw.Draw(canvas)

    left_origin = (16, 16)
    right_origin = (mask.shape[1] + 32, 16)

    _draw_mask(draw, mask, left_origin, fill=(38, 78, 112))
    _draw_contour(draw, result.contour_xy, right_origin, color=(120, 120, 120), width=2)

    palette = [(217, 88, 81), (79, 145, 87), (84, 120, 196), (189, 132, 53)]
    for idx, segment in enumerate(result.contour_segments_xy):
        _draw_contour(draw, segment, right_origin, color=palette[idx % len(palette)], width=3)

    for point_x, point_y in result.minima_points_xy.tolist():
        _draw_point(draw, point_x, point_y, right_origin, color=(235, 179, 24), radius=4)

    centroid_x, centroid_y = result.centroid_xy
    _draw_point(draw, centroid_x, centroid_y, right_origin, color=(20, 20, 20), radius=3)

    draw.text((16, 4), "Synthetic under-segmented overlap mask", fill=(30, 30, 30))
    draw.text((mask.shape[1] + 32, 4), "Contour split at retained local minima", fill=(30, 30, 30))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _draw_mask(draw: ImageDraw.ImageDraw, mask: np.ndarray, origin: tuple[int, int], *, fill: tuple[int, int, int]) -> None:
    top, left = origin[1], origin[0]
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        draw.point((left + col, top + row), fill=fill)


def _draw_contour(
    draw: ImageDraw.ImageDraw,
    contour_xy: np.ndarray,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if contour_xy.shape[0] < 2:
        return
    offset_points = [(origin[0] + int(x), origin[1] + int(y)) for x, y in contour_xy.tolist()]
    draw.line(offset_points + [offset_points[0]], fill=color, width=width)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cx = origin[0] + float(x)
    cy = origin[1] + float(y)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)


if __name__ == "__main__":
    path = render_demo()
    print(path)
