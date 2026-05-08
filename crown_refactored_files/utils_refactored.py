"""Utility functions for crown canopy clustering.

This module keeps the original FamNet helper interface, but separates image
normalization, vegetation masking, exemplar bounding boxes, peak seeding, label
post-processing, and visualization. Coordinates are consistently row/col for
points and [y1, x1, y2, x2] for bounding boxes unless noted otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants used by the original FamNet implementation
# ---------------------------------------------------------------------------

CHANNELS = ("blue", "green", "red", "red_edge", "nir")
FILE_BANDS = ("b", "g", "r", "rEd", "nir")
DATES = ("2020_11_21_1", "2020_11_21_2", "2020_11_22_1")

MAPS = ["map3", "map4"]
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

BBox = List[int]  # [y1, x1, y2, x2]


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Elementwise division with a small denominator guard."""
    return np.asarray(a, dtype=np.float32) / (np.asarray(b, dtype=np.float32) + eps)


def minmax_scale(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale one array to [0, 1], treating non-finite values as invalid."""
    x = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    lo = float(x[finite].min())
    hi = float(x[finite].max())
    if hi - lo <= eps:
        return out

    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def normalize_per_band(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each temporal slice and channel independently."""
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected data shape [T, H, W, C], got {x.shape}")

    out = np.zeros_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        for ch in range(x.shape[-1]):
            out[t, :, :, ch] = minmax_scale(x[t, :, :, ch], eps=eps)
    return out


def normalize_channels(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each channel of a 2D/3D image independently."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected image shape [H, W] or [H, W, C], got {x.shape}")
    out = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[-1]):
        out[..., ch] = minmax_scale(x[..., ch], eps=eps)
    return out


def normalize_to_uint8(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale an array to uint8 [0, 255]."""
    out = minmax_scale(img, eps=eps)
    return np.uint8(np.clip(out, 0.0, 1.0) * 255)


def ensure_3d_image(image: np.ndarray) -> np.ndarray:
    """Return [H, W, C] even when image is [H, W]."""
    x = np.asarray(image)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected image shape [H, W] or [H, W, C], got {x.shape}")


def stack_feature_maps(features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stack an ordered feature dictionary into an [H, W, C] cube."""
    if not features:
        raise ValueError("features cannot be empty")
    return np.stack([np.asarray(v, dtype=np.float32) for v in features.values()], axis=-1)


# Backwards-compatible alias used by the previous pipeline.
stack_feature_dict = stack_feature_maps


# ---------------------------------------------------------------------------
# Bounding-box helpers and interactive exemplar selection
# ---------------------------------------------------------------------------


def validate_bounding_boxes(
    boxes: Sequence[Sequence[int | float]],
    image_shape: Tuple[int, int],
    *,
    clip: bool = True,
    min_size: int = 2,
) -> List[BBox]:
    """Validate and optionally clip [y1, x1, y2, x2] boxes to image bounds."""
    h, w = image_shape
    clean: List[BBox] = []
    for raw in boxes:
        if len(raw) != 4:
            raise ValueError(f"Each bounding box must have 4 values, got {raw}")
        y1, x1, y2, x2 = [int(round(float(v))) for v in raw]
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 < x1:
            x1, x2 = x2, x1
        if clip:
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
        if y2 - y1 + 1 >= min_size and x2 - x1 + 1 >= min_size:
            clean.append([y1, x1, y2, x2])
    return clean


def read_bbox_file(
    bbox_file: str | Path,
    image_shape: Tuple[int, int] | None = None,
    *,
    clip: bool = True,
) -> List[BBox]:
    """Read one [y1 x1 y2 x2] box per line from a text file."""
    path = Path(bbox_file)
    if not path.exists():
        raise FileNotFoundError(f"Bounding-box file not found: {path}")

    boxes: List[List[int]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid bbox line {line_no} in {path}: expected 4 numbers"
                )
            boxes.append([int(round(float(v))) for v in parts])

    if image_shape is not None:
        return validate_bounding_boxes(boxes, image_shape, clip=clip)
    return [list(map(int, b)) for b in boxes]


def write_bbox_file(boxes: Sequence[Sequence[int | float]], out_file: str | Path) -> None:
    """Write [y1 x1 y2 x2] boxes to a text file."""
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for box in boxes:
            y1, x1, y2, x2 = [int(round(float(v))) for v in box]
            fout.write(f"{y1} {x1} {y2} {x2}\n")


def adjust_boxes_for_crop(
    boxes: Sequence[Sequence[int | float]],
    crop: Sequence[int] | None,
    cropped_shape: Tuple[int, int],
    *,
    coordinate_space: str = "full",
) -> List[BBox]:
    """Map full-image boxes into crop coordinates or validate crop-space boxes."""
    if crop is None or coordinate_space == "crop":
        return validate_bounding_boxes(boxes, cropped_shape, clip=True)

    if coordinate_space != "full":
        raise ValueError("coordinate_space must be either 'full' or 'crop'")

    r0, r1, c0, c1 = [int(v) for v in crop]
    shifted: List[BBox] = []
    for y1, x1, y2, x2 in boxes:
        yy1 = int(y1) - r0
        yy2 = int(y2) - r0
        xx1 = int(x1) - c0
        xx2 = int(x2) - c0
        # Keep only boxes that intersect the crop.
        if yy2 < 0 or xx2 < 0 or yy1 >= (r1 - r0) or xx1 >= (c1 - c0):
            continue
        shifted.append([yy1, xx1, yy2, xx2])
    return validate_bounding_boxes(shifted, cropped_shape, clip=True)


def select_exemplar_rois(image: np.ndarray, window_name: str = "image") -> List[BBox]:
    """Interactively select exemplar boxes with OpenCV.

    Press 'n' to add a new ROI and 'q' or Esc to finish. Returned boxes follow
    [y1, x1, y2, x2], matching the original FamNet demo.
    """
    all_rois: List[BBox] = []
    display = np.asarray(image).copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)
    print("Press 'n' to draw a new exemplar. Press 'q' or Esc to finish.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), 13):
            x, y, bw, bh = cv2.selectROI(window_name, display, False, False)
            if bw <= 0 or bh <= 0:
                continue
            y1, x1, y2, x2 = y, x, y + bh - 1, x + bw - 1
            all_rois.append([int(y1), int(x1), int(y2), int(x2)])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow(window_name, display)
            print(f"Added ROI: {[int(y1), int(x1), int(y2), int(x2)]}")

    cv2.destroyWindow(window_name)
    return all_rois


def select_or_load_exemplar_boxes(
    image_rgb: np.ndarray,
    *,
    bbox_file: str | Path | None = None,
    interactive: bool = False,
    output_bbox_file: str | Path | None = None,
) -> List[BBox]:
    """Load boxes from file or let the user draw them interactively."""
    h, w = image_rgb.shape[:2]
    if bbox_file is not None:
        boxes = read_bbox_file(bbox_file, image_shape=(h, w), clip=True)
    elif interactive:
        display = np.uint8(np.clip(image_rgb, 0.0, 1.0) * 255)
        # PIL/NumPy RGB to OpenCV BGR for display.
        boxes = select_exemplar_rois(cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        boxes = validate_bounding_boxes(boxes, (h, w), clip=True)
    else:
        boxes = []

    if boxes and output_bbox_file is not None:
        write_bbox_file(boxes, output_bbox_file)
    return boxes


def boxes_to_centers_rc(boxes: Sequence[Sequence[int | float]]) -> np.ndarray:
    """Convert boxes to center points in row/col coordinates."""
    centers = []
    for y1, x1, y2, x2 in boxes:
        centers.append([(float(y1) + float(y2)) / 2.0, (float(x1) + float(x2)) / 2.0])
    return np.rint(np.asarray(centers, dtype=np.float32)).astype(np.int32)


def box_radius_estimates(
    boxes: Sequence[Sequence[int | float]],
    *,
    scale: float = 0.65,
    min_radius: float = 8.0,
    max_radius: float | None = None,
) -> np.ndarray:
    """Estimate one crown-support radius per box from its diagonal."""
    radii: List[float] = []
    for y1, x1, y2, x2 in boxes:
        height = max(1.0, float(y2) - float(y1) + 1.0)
        width = max(1.0, float(x2) - float(x1) + 1.0)
        r = scale * 0.5 * math.sqrt(height * height + width * width)
        r = max(float(min_radius), r)
        if max_radius is not None:
            r = min(float(max_radius), r)
        radii.append(r)
    return np.asarray(radii, dtype=np.float32)


# ---------------------------------------------------------------------------
# FamNet loss, feature extraction, and inference helpers
# ---------------------------------------------------------------------------


def matlab_style_gauss2D(shape: Tuple[int, int] = (3, 3), sigma: float = 0.5) -> np.ndarray:
    """2D Gaussian mask compatible with MATLAB fspecial('gaussian')."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def _iter_famnet_boxes(boxes: torch.Tensor) -> Iterable[torch.Tensor]:
    boxes = boxes.squeeze()
    if boxes.ndim == 1:
        yield boxes
    else:
        for box in boxes:
            yield box


def PerturbationLoss(output: torch.Tensor, boxes: torch.Tensor, sigma: int = 8, use_gpu: bool = True):
    """Original FamNet perturbation loss, with safer box iteration."""
    loss = 0.0
    for temp_box in _iter_famnet_boxes(boxes):
        y1 = int(temp_box[1])
        y2 = int(temp_box[3])
        x1 = int(temp_box[2])
        x2 = int(temp_box[4])
        out = output[:, :, y1:y2, x1:x2]
        if out.numel() == 0:
            continue
        gauss = matlab_style_gauss2D(shape=(out.shape[2], out.shape[3]), sigma=sigma)
        kernel = torch.from_numpy(gauss).float().to(output.device if use_gpu else "cpu")
        loss += F.mse_loss(out.squeeze(), kernel)
    return loss


def MincountLoss(output: torch.Tensor, boxes: torch.Tensor, use_gpu: bool = True):
    """Original FamNet min-count loss, with safer box iteration."""
    ones = torch.ones(1, device=output.device if use_gpu else "cpu")
    loss = 0.0
    for temp_box in _iter_famnet_boxes(boxes):
        y1 = int(temp_box[1])
        y2 = int(temp_box[3])
        x1 = int(temp_box[2])
        x2 = int(temp_box[4])
        roi_sum = output[:, :, y1:y2, x1:x2].sum()
        if roi_sum.item() <= 1:
            loss += F.mse_loss(roi_sum, ones)
    return loss


class resizeImage(object):
    """Resize PIL image while preserving aspect ratio and scaling boxes."""

    def __init__(self, MAX_HW: int = MAX_HW):
        self.max_hw = int(MAX_HW)

    def __call__(self, sample: MutableMapping[str, object]) -> Dict[str, object]:
        image = sample["image"]
        lines_boxes = sample["lines_boxes"]
        if not isinstance(image, Image.Image):
            raise TypeError("sample['image'] must be a PIL.Image")

        width, height = image.size
        if width > self.max_hw or height > self.max_hw:
            scale_factor = float(self.max_hw) / max(height, width)
            new_h = max(8, 8 * int(height * scale_factor / 8))
            new_w = max(8, 8 * int(width * scale_factor / 8))
            resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            scale_factor = 1.0
            resized_image = image

        boxes = []
        for box in lines_boxes:  # type: ignore[union-attr]
            y1, x1, y2, x2 = [int(round(float(k) * scale_factor)) for k in box]
            boxes.append([0, y1, x1, y2, x2])
        if len(boxes) == 0:
            raise ValueError("At least one exemplar box is required for FamNet")

        resized_image_t = Normalize(resized_image)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)
        return {"image": resized_image_t, "boxes": boxes_t}


class resizeImageWithGT(object):
    """Training transform retained for compatibility with the original project."""

    def __init__(self, MAX_HW: int = MAX_HW):
        self.max_hw = int(MAX_HW)

    def __call__(self, sample: MutableMapping[str, object]) -> Dict[str, object]:
        image = sample["image"]
        lines_boxes = sample["lines_boxes"]
        density = np.asarray(sample["gt_density"], dtype=np.float32)
        if not isinstance(image, Image.Image):
            raise TypeError("sample['image'] must be a PIL.Image")

        width, height = image.size
        if width > self.max_hw or height > self.max_hw:
            scale_factor = float(self.max_hw) / max(height, width)
            new_h = max(8, 8 * int(height * scale_factor / 8))
            new_w = max(8, 8 * int(width * scale_factor / 8))
            resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_density = cv2.resize(density, (new_w, new_h))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)
            if new_count > 0:
                resized_density = resized_density * (orig_count / new_count)
        else:
            scale_factor = 1.0
            resized_image = image
            resized_density = density

        boxes = []
        for box in lines_boxes:  # type: ignore[union-attr]
            y1, x1, y2, x2 = [int(round(float(k) * scale_factor)) for k in box]
            boxes.append([0, y1, x1, y2, x2])

        return {
            "image": Normalize(resized_image),
            "boxes": torch.tensor(boxes, dtype=torch.float32).unsqueeze(0),
            "gt_density": torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0),
        }


class NormalizeImage(object):
    """Convert a PIL RGB image to an ImageNet-normalized CHW tensor."""

    def __init__(self, mean=IM_NORM_MEAN, std=IM_NORM_STD):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


class Compose(object):
    """Small local replacement for torchvision.transforms.Compose."""

    def __init__(self, transforms_seq):
        self.transforms_seq = list(transforms_seq)

    def __call__(self, sample):
        out = sample
        for transform in self.transforms_seq:
            out = transform(out)
        return out


Normalize = NormalizeImage()
Transform = Compose([resizeImage(MAX_HW)])
TransformTrain = Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor: torch.Tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD) -> torch.Tensor:
    """Reverse ImageNet normalization for plotting."""
    x = tensor.clone()
    if x.ndim == 4:
        x = x.squeeze(0)
    for channel, mean, std in zip(x, means, stds):
        channel.mul_(std).add_(mean)
    return x


def format_for_plotting(tensor):
    """Convert CxHxW or NxCxHxW tensor to HxWxC or HxW for plotting."""
    if torch.is_tensor(tensor):
        formatted = tensor.detach().cpu()
        if formatted.ndim == 4:
            formatted = formatted.squeeze(0)
        if formatted.shape[0] == 1:
            return formatted.squeeze(0)
        return formatted.permute(1, 2, 0)
    arr = np.asarray(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            return arr[0]
        return np.moveaxis(arr, 0, -1)
    return arr


def extract_features(
    feature_model,
    image: torch.Tensor,
    boxes: torch.Tensor,
    feat_map_keys: Sequence[str] = MAPS,
    exemplar_scales: Sequence[float] = Scales,
):
    """Extract convolutional exemplar-response features for FamNet."""
    n_images = image.shape[0]
    n_boxes = boxes.shape[2]
    image_features = feature_model(image)

    all_features = []
    for ix in range(n_images):
        boxes_ix = boxes[ix][0]
        combined_by_level = []
        for key in feat_map_keys:
            feat = image_features[key][ix].unsqueeze(0)
            if key in ("map1", "map2"):
                scaling = 4.0
            elif key == "map3":
                scaling = 8.0
            elif key == "map4":
                scaling = 16.0
            else:
                scaling = 32.0

            boxes_scaled = boxes_ix / scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5]) + 1
            feat_h, feat_w = feat.shape[-2], feat.shape[-1]
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
            max_h = max(1, math.ceil(float(torch.max(box_hs).item())))
            max_w = max(1, math.ceil(float(torch.max(box_ws).item())))

            examples = []
            for j in range(n_boxes):
                y1, x1 = int(boxes_scaled[j, 1]), int(boxes_scaled[j, 2])
                y2, x2 = int(boxes_scaled[j, 3]), int(boxes_scaled[j, 4])
                crop = feat[:, :, y1:y2, x1:x2]
                if crop.shape[2] == 0 or crop.shape[3] == 0:
                    crop = torch.zeros(
                        (1, feat.shape[1], max_h, max_w),
                        dtype=feat.dtype,
                        device=feat.device,
                    )
                elif crop.shape[2] != max_h or crop.shape[3] != max_w:
                    crop = F.interpolate(crop, size=(max_h, max_w), mode="bilinear")
                examples.append(crop)
            example_features = torch.cat(examples, dim=0)

            h, w = example_features.shape[2], example_features.shape[3]
            level_responses = []
            for scale in [1.0, *exemplar_scales]:
                if abs(scale - 1.0) < 1e-12:
                    examples_scaled = example_features
                    h_s, w_s = h, w
                else:
                    h_s = max(1, math.ceil(h * float(scale)))
                    w_s = max(1, math.ceil(w * float(scale)))
                    examples_scaled = F.interpolate(
                        example_features, size=(h_s, w_s), mode="bilinear"
                    )
                response = F.conv2d(
                    F.pad(feat, (int(w_s / 2), int((w_s - 1) / 2), int(h_s / 2), int((h_s - 1) / 2))),
                    examples_scaled,
                ).permute([1, 0, 2, 3])
                level_responses.append(response)

            combined = torch.cat(level_responses, dim=1)
            if combined_by_level and (
                combined_by_level[0].shape[2] != combined.shape[2]
                or combined_by_level[0].shape[3] != combined.shape[3]
            ):
                combined = F.interpolate(
                    combined,
                    size=(combined_by_level[0].shape[2], combined_by_level[0].shape[3]),
                    mode="bilinear",
                )
            combined_by_level.append(combined)
        all_features.append(torch.cat(combined_by_level, dim=1).unsqueeze(0))
    return torch.cat(all_features, dim=0)


def adapt_regressor(regressor, features: torch.Tensor, boxes: torch.Tensor, args, device: torch.device):
    """Optional test-time adaptation for FamNet."""
    regressor.train()
    optimizer = optim.Adam(regressor.parameters(), lr=float(args.learning_rate))
    use_gpu = device.type != "cpu"

    for _ in tqdm(range(int(args.gradient_steps)), desc="Adapting"):
        optimizer.zero_grad(set_to_none=True)
        output = regressor(features)
        loss_count = float(args.weight_mincount) * MincountLoss(output, boxes, use_gpu=use_gpu)
        loss_perturb = float(args.weight_perturbation) * PerturbationLoss(
            output, boxes, sigma=8, use_gpu=use_gpu
        )
        loss = loss_count + loss_perturb
        if torch.is_tensor(loss):
            loss.backward()
            optimizer.step()
    regressor.eval()
    return regressor


def run_famnet(
    feature_image: np.ndarray,
    boxes: Sequence[Sequence[int | float]],
    resnet,
    regressor,
    load_density: bool,
    args,
    device: torch.device,
    *,
    cache_name: str | None = None,
) -> np.ndarray:
    """Run FamNet on one scalar feature image using exemplar boxes."""
    if len(boxes) == 0:
        raise RuntimeError("No exemplar boxes found; cannot run FamNet.")

    output_dir = Path(args.output_dir)
    cache_name = cache_name or f"density_day{getattr(args, 'day_for_ranking', 0)}_adapt{int(args.adapt)}.pt"
    cache_path = output_dir / cache_name
    if load_density and cache_path.exists():
        output = torch.load(cache_path, map_location=device)
        out = format_for_plotting(output)
        if torch.is_tensor(out):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float32)

    img = minmax_scale(feature_image)
    img_3ch = np.stack([img, img, img], axis=-1)
    pil_img = Image.fromarray(np.uint8(np.clip(img_3ch, 0.0, 1.0) * 255))
    sample = Transform({"image": pil_img, "lines_boxes": [list(map(int, b)) for b in boxes]})
    t_image = sample["image"].unsqueeze(0).to(device)
    t_boxes = sample["boxes"].unsqueeze(0).to(device)

    with torch.no_grad():
        features = extract_features(resnet, t_image, t_boxes, MAPS, Scales)

    if bool(args.adapt):
        features.requires_grad_(True)
        regressor = adapt_regressor(regressor, features, t_boxes, args, device)

    with torch.no_grad():
        output = regressor(features).detach().cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cache_path)

    out = format_for_plotting(output)
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Vegetation indices and RGB-specific features
# ---------------------------------------------------------------------------


def compute_vegetation_indices(xn: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute multispectral vegetation indices from [T, H, W, >=5]."""
    x = np.asarray(xn, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] < 5:
        raise ValueError(f"Expected shape [T, H, W, >=5], got {x.shape}")

    b, g, r, re, nir = (x[..., i] for i in range(5))
    msavi_disc = np.maximum((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - r), 0.0)
    indices = {
        "NDVI": safe_div(nir - r, nir + r),
        "SAVI": 1.5 * safe_div(nir - r, nir + r + 0.5),
        "OSAVI": 1.16 * safe_div(nir - r, nir + r + 0.16),
        "MSAVI": (2.0 * nir + 1.0 - np.sqrt(msavi_disc)) / 2.0,
        "DVI": nir - r,
        "NDRE": safe_div(nir - re, nir + re),
        "GNDVI": safe_div(nir - g, nir + g),
        "NRI": safe_div(r, r + g + b),
        "VARI": safe_div(g - r, g + r - b),
        "PPR": safe_div(g - b, g + b),
        "ARVI": safe_div(nir - (2.0 * r - b), nir + (2.0 * r - b)),
    }
    return {name: minmax_scale(values) for name, values in indices.items()}


def compute_rgb_indices(xn: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute RGB vegetation proxies from [T, H, W, 3] RGB in [0, 1]."""
    x = np.asarray(xn, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected shape [T, H, W, 3], got {x.shape}")
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    indices = {
        "VARI": safe_div(g - r, g + r - b),
        "PPR": safe_div(g - b, g + b),
        "ExG": 2.0 * g - r - b,
        "GLI": safe_div(2.0 * g - r - b, 2.0 * g + r + b),
        "ExR": 1.4 * r - g,
    }
    return {name: minmax_scale(values) for name, values in indices.items()}


def build_candidate_features(xn: np.ndarray, indices: Mapping[str, np.ndarray] | None = None, day: int = 0) -> Dict[str, np.ndarray]:
    """Build candidate feature maps for multispectral or RGB image stacks."""
    x = np.asarray(xn, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected shape [T, H, W, C], got {x.shape}")
    day = int(np.clip(day, 0, x.shape[0] - 1))

    if x.shape[-1] >= 5:
        band_names = ("Blue", "Green", "Red", "RedEdge", "NIR")
        out = {name: x[day, :, :, i] for i, name in enumerate(band_names)}
        idx = indices if indices is not None else compute_vegetation_indices(x)
        out.update({name: values[day] for name, values in idx.items()})
        return out

    if x.shape[-1] == 3:
        out = {"Red": x[day, :, :, 0], "Green": x[day, :, :, 1], "Blue": x[day, :, :, 2]}
        idx = indices if indices is not None else compute_rgb_indices(x)
        out.update({name: values[day] for name, values in idx.items()})
        return out

    raise ValueError(f"Unsupported number of channels: {x.shape[-1]}")


def compute_rgb_vegetation_mask(
    rgb_image: np.ndarray,
    *,
    use_otsu: bool = True,
    threshold: float = 0.35,
    min_fraction_fallback: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a soft RGB vegetation mask using ExG and VARI.

    RGB masks are less reliable than multispectral masks. If the mask is nearly
    empty, the function falls back to all finite pixels so that downstream crown
    delineation can still run from manual exemplars.
    """
    rgb = np.asarray(rgb_image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape [H, W, 3], got {rgb.shape}")
    rgb = normalize_channels(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    exg = 2.0 * g - r - b
    vari = safe_div(g - r, g + r - b)
    gli = safe_div(2.0 * g - r - b, 2.0 * g + r + b)
    score = 0.55 * minmax_scale(exg) + 0.30 * minmax_scale(vari) + 0.15 * minmax_scale(gli)
    score8 = np.uint8(np.clip(score, 0.0, 1.0) * 255)

    if use_otsu:
        _, mask = cv2.threshold(score8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (score > float(threshold)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    keep = mask > 0

    finite = np.all(np.isfinite(rgb), axis=-1)
    if keep.mean() < float(min_fraction_fallback):
        keep = finite
    else:
        keep &= finite
    return keep.astype(np.uint8), score


def compute_shadow_removal_mask(
    x: np.ndarray,
    vegetation_mask: np.ndarray | None = None,
    shadow_percentile: float = 15.0,
    nir_percentile: float = 10.0,
    rededge_percentile: float = 10.0,
    use_morphology: bool = True,
    max_shadow_fraction: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, np.ndarray | float]]:
    """Remove likely shadow pixels from a multispectral vegetation mask.

    The previous implementation could erase an entire vegetation mask when the
    candidate vegetation pixels had nearly constant brightness/NIR/RedEdge values
    because every pixel was equal to the low-percentile thresholds. This guarded
    version skips shadow removal when the valid vegetation distribution is too
    flat, and falls back to the original vegetation mask if the shadow test would
    remove an implausibly large fraction of vegetation.
    """
    img = np.asarray(x, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] < 5:
        raise ValueError(f"Expected shape [H, W, >=5], got {img.shape}")

    blue, green, red, red_edge, nir = (img[..., i] for i in range(5))
    visible_brightness = (blue + green + red) / 3.0
    finite = np.isfinite(visible_brightness) & np.isfinite(nir) & np.isfinite(red_edge)
    veg = np.ones(img.shape[:2], dtype=bool) if vegetation_mask is None else np.asarray(vegetation_mask, dtype=bool)
    valid = finite & veg
    if not np.any(valid):
        keep = np.zeros(img.shape[:2], dtype=np.uint8)
        return keep, {"shadow_mask": np.ones(img.shape[:2], dtype=bool), "vegetation_mask": veg}

    valid_vis = visible_brightness[valid]
    valid_nir = nir[valid]
    valid_re = red_edge[valid]
    debug_base: Dict[str, np.ndarray | float] = {
        "visible_brightness": visible_brightness,
        "vegetation_mask": veg,
    }

    dynamic_range = max(
        float(np.ptp(valid_vis)),
        float(np.ptp(valid_nir)),
        float(np.ptp(valid_re)),
    )
    if dynamic_range <= 1e-6:
        shadow = np.zeros(img.shape[:2], dtype=bool)
        return (veg & finite).astype(np.uint8), {
            **debug_base,
            "shadow_mask": shadow,
            "shadow_skip_reason": np.asarray("flat_valid_distribution", dtype=object),
        }

    vis_thr = float(np.percentile(valid_vis, shadow_percentile))
    nir_thr = float(np.percentile(valid_nir, nir_percentile))
    re_thr = float(np.percentile(valid_re, rededge_percentile))

    # Strict comparisons avoid classifying all pixels as shadow when many values
    # are tied exactly at the percentile threshold.
    shadow = (visible_brightness < vis_thr) & (nir < nir_thr) & (red_edge < re_thr) & valid
    shadow_fraction = float(shadow.sum() / max(int(valid.sum()), 1))

    if shadow_fraction > float(max_shadow_fraction):
        shadow = np.zeros(img.shape[:2], dtype=bool)
        keep_mask = veg & finite
    else:
        keep_mask = veg & finite & (~shadow)

    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep_u8 = cv2.morphologyEx(keep_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, kernel)
        keep_mask = keep_u8.astype(bool)

    return keep_mask.astype(np.uint8), {
        **debug_base,
        "shadow_mask": shadow,
        "vis_thr": vis_thr,
        "nir_thr": nir_thr,
        "rededge_thr": re_thr,
        "shadow_fraction": shadow_fraction,
    }


def compute_ground_removal_mask(
    x: np.ndarray,
    method: str = "combined",
    use_otsu: bool = True,
    remove_shadow: bool = True,
    shadow_percentile: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a multispectral vegetation mask from NDVI/OSAVI/MSAVI/NDRE."""
    img = np.asarray(x, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] < 5:
        raise ValueError(f"Expected shape [H, W, >=5], got {img.shape}")

    _, _, red, red_edge, nir = (img[..., i] for i in range(5))
    ndvi = safe_div(nir - red, nir + red)
    osavi = 1.16 * safe_div(nir - red, nir + red + 0.16)
    msavi_term = np.maximum((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red), 0.0)
    msavi = (2.0 * nir + 1.0 - np.sqrt(msavi_term)) / 2.0
    ndre = safe_div(nir - red_edge, nir + red_edge)

    if method == "ndvi":
        index = ndvi
    elif method == "osavi":
        index = osavi
    elif method == "msavi":
        index = msavi
    elif method == "combined":
        index = 0.45 * ndvi + 0.35 * osavi + 0.20 * ndre
    else:
        raise ValueError("method must be one of: ndvi, osavi, msavi, combined")

    index8 = normalize_to_uint8(index)
    if use_otsu:
        _, mask = cv2.threshold(index8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (index > 0.2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    vegetation_mask = (mask > 0).astype(np.uint8)

    if remove_shadow:
        vegetation_mask, _ = compute_shadow_removal_mask(
            img, vegetation_mask=vegetation_mask, shadow_percentile=shadow_percentile
        )
    return vegetation_mask.astype(np.uint8), index


def compute_scene_vegetation_mask(
    scene_image: np.ndarray,
    input_mode: str,
    ground_method: str = "combined",
    *,
    mask_mode: str = "auto",
    remove_shadow: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified vegetation mask for multispectral or RGB inputs."""
    img = np.asarray(scene_image, dtype=np.float32)
    if mask_mode == "all":
        return np.ones(img.shape[:2], dtype=np.uint8), np.ones(img.shape[:2], dtype=np.float32)
    if mask_mode != "auto":
        raise ValueError("mask_mode must be 'auto' or 'all'")

    if input_mode == "multispectral":
        return compute_ground_removal_mask(
            img[:, :, :5], method=ground_method, use_otsu=True, remove_shadow=remove_shadow
        )
    if input_mode == "rgb":
        return compute_rgb_vegetation_mask(img[:, :, :3], use_otsu=True)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


# ---------------------------------------------------------------------------
# Candidate ranking and automatic exemplar blobs
# ---------------------------------------------------------------------------


def mask_bright_spots(
    candidates: Mapping[str, np.ndarray],
    laplacian_ksize: int = 3,
    dilate_ksize: int = 5,
) -> Dict[str, np.ndarray]:
    """Detect bright/high-frequency candidate blobs in each feature map."""
    kernel = np.ones((int(dilate_ksize), int(dilate_ksize)), dtype=np.uint8)
    masks: Dict[str, np.ndarray] = {}
    for name, img in candidates.items():
        gray = normalize_to_uint8(img)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=int(laplacian_ksize))
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        dilated = cv2.dilate(lap, kernel, iterations=1)
        _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks[name] = thresh
    return masks


def compute_feature_metrics(feature_img: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """Score how well a candidate map separates detected bright blobs."""
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
    pooled_std = math.sqrt((std_in * std_in + std_out * std_out) / 2.0)
    return {
        **base,
        "contrast_ratio": (mu_in - mu_out) / (abs(mu_out) + eps),
        "effect_size": (mu_in - mu_out) / (pooled_std + eps),
        "fisher_score": (mu_in - mu_out) ** 2 / (std_in * std_in + std_out * std_out + eps),
    }


def rank_candidates(
    candidates: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    sort_by: str = "contrast_ratio",
) -> List[Dict[str, float | str]]:
    """Rank candidate maps by a contrast metric."""
    rows: List[Dict[str, float | str]] = []
    for name, img in candidates.items():
        if name not in masks:
            continue
        row: Dict[str, float | str] = compute_feature_metrics(img, masks[name])
        row["name"] = name
        rows.append(row)
    rows.sort(key=lambda x: float(x.get(sort_by, float("-inf"))), reverse=True)
    return rows


def detect_big_round_blobs(
    mask: np.ndarray,
    min_area: float = 50.0,
    min_circularity: float = 0.6,
    min_solidity: float = 0.85,
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """Keep connected components that look like compact round crown exemplars."""
    bw = (np.asarray(mask) > 0).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept_mask = np.zeros_like(bw)
    info: List[dict] = []
    kept: List[np.ndarray] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < float(min_circularity):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < float(min_solidity):
            continue
        kept.append(cnt)
        cv2.drawContours(kept_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        info.append({"area": area, "circularity": circularity, "solidity": solidity})
    return kept_mask, info, kept


def top_k_blobs(blob_info: Sequence[dict], contours: Sequence[np.ndarray], k: int) -> Tuple[List[dict], List[np.ndarray]]:
    """Return the k highest area*circularity blobs."""
    if not blob_info or not contours:
        return [], []
    scores = np.asarray([float(b["area"]) * float(b["circularity"]) for b in blob_info], dtype=np.float32)
    idx = np.argsort(scores)[::-1][: int(k)]
    return [blob_info[i] for i in idx], [contours[i] for i in idx]


def filter_border_contours(
    shape: Tuple[int, int],
    contours: Sequence[np.ndarray],
    margin: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Discard contours touching the image border."""
    h, w = shape
    kept: List[np.ndarray] = []
    out = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue
        kept.append(cnt)
    if kept:
        cv2.drawContours(out, kept, -1, 255, thickness=cv2.FILLED)
    return out, kept


def get_blob_bounding_boxes(contours: Sequence[np.ndarray]) -> Tuple[List[BBox], float]:
    """Convert contours to [y1, x1, y2, x2] boxes and return min box area."""
    boxes: List[BBox] = []
    areas: List[float] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([int(y), int(x), int(y + h - 1), int(x + w - 1)])
        areas.append(float(w * h))
    return boxes, float(min(areas)) if areas else 0.0


# ---------------------------------------------------------------------------
# Peaks, seeding, supports, and label post-processing
# ---------------------------------------------------------------------------


def detect_strong_peaks(
    image: np.ndarray,
    min_distance: int = 5,
    percentile: float = 80.0,
    exclude_border: bool = True,
) -> np.ndarray:
    """Detect density/local-maxima peaks in row/col coordinates."""
    x = np.asarray(image, dtype=np.float32)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return np.empty((0, 2), dtype=np.int32)
    threshold = float(np.percentile(x[np.isfinite(x)], percentile))
    peaks = peak_local_max(
        x,
        min_distance=int(min_distance),
        threshold_abs=threshold,
        exclude_border=exclude_border,
    )
    return np.asarray(peaks, dtype=np.int32)


def sample_points_in_circle_xy(
    center_xy: Tuple[float, float],
    area: float,
    num_points: int,
    image_shape: Tuple[int, int],
    seed: int | None = None,
) -> np.ndarray:
    """Sample points inside a circle. Returns points in x/y order."""
    rng = np.random.default_rng(seed)
    cx, cy = center_xy
    radius = math.sqrt(max(float(area), 1.0) / math.pi)
    h, w = image_shape
    pts_out: List[List[float]] = []
    max_trials = 1000
    trials = 0
    while len(pts_out) < int(num_points) and trials < max_trials:
        trials += 1
        n = max(16, 3 * (int(num_points) - len(pts_out)))
        r = radius * np.sqrt(rng.random(n))
        theta = 2.0 * math.pi * rng.random(n)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        pts_out.extend(np.column_stack((xs[valid], ys[valid])).tolist())
    if not pts_out:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts_out[: int(num_points)], dtype=np.float32)


def xy_to_rc(points_xy: np.ndarray) -> np.ndarray:
    """Convert points from x/y to row/col integer coordinates."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    return np.column_stack((pts[:, 1], pts[:, 0])).astype(np.int32)


def sample_seed_clusters_from_peaks(
    peaks_rc: np.ndarray,
    radii: Sequence[int | float],
    image_shape: Tuple[int, int],
    vegetation_mask: np.ndarray | None,
    *,
    num_points: int = 30,
    radius_fraction: float = 0.45,
    min_seed_radius: float = 3.0,
    random_seed: int | None = 12345,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Create one sampled seed cluster per peak."""
    peaks = np.asarray(peaks_rc, dtype=np.int32).reshape(-1, 2)
    h, w = image_shape
    if len(peaks) == 0:
        return [], np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    radii_arr = np.asarray(radii, dtype=np.float32).reshape(-1)
    if len(radii_arr) == 1 and len(peaks) > 1:
        radii_arr = np.full(len(peaks), float(radii_arr[0]), dtype=np.float32)
    if len(radii_arr) != len(peaks):
        raise ValueError(f"Expected {len(peaks)} radii, got {len(radii_arr)}")

    veg = np.ones((h, w), dtype=bool) if vegetation_mask is None else np.asarray(vegetation_mask, dtype=bool)
    clusters: List[np.ndarray] = []
    valid_peaks: List[List[int]] = []
    valid_radii: List[float] = []
    rng = np.random.default_rng(random_seed)

    for peak, radius in zip(peaks, radii_arr):
        row, col = int(peak[0]), int(peak[1])
        if not (0 <= row < h and 0 <= col < w):
            continue
        seed_radius = max(float(min_seed_radius), float(radius) * float(radius_fraction))
        area = math.pi * seed_radius * seed_radius
        pts_xy = sample_points_in_circle_xy(
            (float(col), float(row)), area, int(num_points), (h, w), seed=int(rng.integers(0, 2**31 - 1))
        )
        pts_rc = xy_to_rc(pts_xy)
        if pts_rc.size == 0:
            pts_rc = np.asarray([[row, col]], dtype=np.int32)
        pts_rc[:, 0] = np.clip(pts_rc[:, 0], 0, h - 1)
        pts_rc[:, 1] = np.clip(pts_rc[:, 1], 0, w - 1)
        keep = veg[pts_rc[:, 0], pts_rc[:, 1]]
        pts_rc = np.unique(pts_rc[keep], axis=0)
        if len(pts_rc) == 0 and veg[row, col]:
            pts_rc = np.asarray([[row, col]], dtype=np.int32)
        if len(pts_rc) == 0:
            continue
        clusters.append(pts_rc.astype(np.int32, copy=False))
        valid_peaks.append([row, col])
        valid_radii.append(float(radius))

    return clusters, np.asarray(valid_peaks, dtype=np.int32), np.asarray(valid_radii, dtype=np.float32)


def make_circle_mask(shape: Tuple[int, int], center_rc: Sequence[int | float], radius: int | float) -> np.ndarray:
    """Create a boolean disk mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    r, c = int(round(float(center_rc[0]))), int(round(float(center_rc[1])))
    cv2.circle(mask, (c, r), int(round(float(radius))), 1, thickness=-1)
    return mask.astype(bool)


def compute_overlap_counts(circle_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Count how many circles cover each pixel."""
    if not circle_masks:
        raise ValueError("circle_masks must not be empty")
    return np.stack(circle_masks, axis=0).astype(np.uint8).sum(axis=0)


def grow_peak_circles_until_collision(
    peaks_rc: np.ndarray,
    image: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
    ground_method: str = "combined",
    remove_shadow: bool = True,
    shadow_percentile: float = 15.0,
    max_intersection_frac: float = 0.10,
    radius_step: int = 2,
    initial_radius: int = 2,
    max_radius: int | None = None,
    use_otsu: bool = True,
) -> Tuple[np.ndarray, List[dict], Dict[str, object]]:
    """Grow one circular support per peak until it hits ground/shadow or neighbors."""
    peaks = np.asarray(peaks_rc, dtype=np.int32).reshape(-1, 2)
    if vegetation_mask is None:
        if image is None:
            raise ValueError("Either vegetation_mask or image must be provided")
        vegetation_mask, vegetation_index = compute_ground_removal_mask(
            np.asarray(image, dtype=np.float32)[..., :5],
            method=ground_method,
            use_otsu=use_otsu,
            remove_shadow=remove_shadow,
            shadow_percentile=shadow_percentile,
        )
    else:
        vegetation_index = None

    veg = np.asarray(vegetation_mask, dtype=bool)
    h, w = veg.shape
    k = len(peaks)
    if k == 0:
        return np.zeros((h, w), dtype=np.int32), [], {
            "vegetation_mask": veg,
            "non_vegetation_mask": ~veg,
            "vegetation_index": vegetation_index,
            "history": [],
        }
    if not (0.0 <= float(max_intersection_frac) <= 1.0):
        raise ValueError("max_intersection_frac must be between 0 and 1")
    if max_radius is None:
        max_radius = int(np.hypot(h, w))

    nonveg = ~veg
    radii = np.full(k, int(initial_radius), dtype=np.int32)
    active = np.ones(k, dtype=bool)
    stop_reasons = ["active"] * k
    history: List[List[dict]] = []

    while np.any(active):
        proposed = radii.copy()
        proposed[active] += int(radius_step)
        masks = [make_circle_mask((h, w), peaks[i], proposed[i]) for i in range(k)]
        overlaps = compute_overlap_counts(masks)
        stop_now = np.zeros(k, dtype=bool)
        records: List[dict] = []

        for i in range(k):
            if not active[i]:
                continue
            circle = masks[i]
            area = int(circle.sum())
            if area == 0:
                stop_now[i] = True
                stop_reasons[i] = "empty_circle"
                continue
            nonveg_frac = float((circle & nonveg).sum() / max(area, 1))
            overlap_frac = float((circle & (overlaps > 1)).sum() / max(area, 1))
            if nonveg_frac >= float(max_intersection_frac):
                stop_now[i] = True
                stop_reasons[i] = "non_vegetation_intersection"
            elif overlap_frac >= float(max_intersection_frac):
                stop_now[i] = True
                stop_reasons[i] = "circle_intersection"
            elif proposed[i] >= int(max_radius):
                stop_now[i] = True
                stop_reasons[i] = "max_radius"
            records.append({
                "circle_id": int(i + 1),
                "radius": int(proposed[i]),
                "area": area,
                "nonveg_frac": nonveg_frac,
                "overlap_frac": overlap_frac,
                "stop": bool(stop_now[i]),
                "reason": stop_reasons[i] if stop_now[i] else "active",
            })

        for i in range(k):
            if active[i] and not stop_now[i]:
                radii[i] = proposed[i]
        active[stop_now] = False
        history.append(records)
        if np.all(proposed >= int(max_radius)):
            break

    final_masks = [make_circle_mask((h, w), peaks[i], radii[i]) for i in range(k)]
    labels = np.zeros((h, w), dtype=np.int32)
    for i, mask in enumerate(final_masks, start=1):
        labels[(mask & veg) & (labels == 0)] = i

    final_overlaps = compute_overlap_counts(final_masks)
    circle_info: List[dict] = []
    for i, mask in enumerate(final_masks):
        area = int(mask.sum())
        circle_info.append({
            "circle_id": int(i + 1),
            "center_rc": tuple(map(int, peaks[i])),
            "radius": int(radii[i]),
            "area": area,
            "vegetated_area": int((mask & veg).sum()),
            "nonveg_frac": float((mask & nonveg).sum() / max(area, 1)),
            "overlap_frac": float((mask & (final_overlaps > 1)).sum() / max(area, 1)),
            "stop_reason": stop_reasons[i],
        })

    return labels, circle_info, {
        "vegetation_mask": veg,
        "non_vegetation_mask": nonveg,
        "vegetation_index": vegetation_index,
        "circle_masks": final_masks,
        "history": history,
    }


def build_cluster_neighborhood_masks(
    image_shape: Tuple[int, int],
    seed_clusters_rc: Sequence[np.ndarray],
    neighborhood_radius: int | float | None = None,
    neighborhood_radii: Sequence[int | float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build one allowed disk-union support mask per seed cluster."""
    h, w = image_shape
    k = len(seed_clusters_rc)
    masks = np.zeros((k, h, w), dtype=np.uint8)
    if k == 0:
        return masks.astype(bool)

    if neighborhood_radii is not None:
        radii = np.asarray(neighborhood_radii, dtype=np.float32).reshape(-1)
        if len(radii) == 1:
            radii = np.full(k, float(radii[0]), dtype=np.float32)
        if len(radii) != k:
            raise ValueError(f"Expected 1 or {k} neighborhood radii, got {len(radii)}")
    else:
        if neighborhood_radius is None:
            masks[:, :, :] = 1
            return masks.astype(bool)
        radii = np.full(k, float(neighborhood_radius), dtype=np.float32)

    for i, cluster in enumerate(seed_clusters_rc):
        radius = int(round(float(radii[i])))
        if radius <= 0:
            masks[i, :, :] = 1
            continue
        pts = np.asarray(cluster, dtype=np.int32).reshape(-1, 2)
        for row, col in pts:
            if 0 <= int(row) < h and 0 <= int(col) < w:
                cv2.circle(masks[i], (int(col), int(row)), radius, 1, thickness=-1)
    return masks.astype(bool)


def reduce_feature_cube(
    image: np.ndarray,
    n_components: int = 3,
    valid_mask: np.ndarray | None = None,
    method: str = "pca",
) -> np.ndarray:
    """Reduce feature channels with PCA/SVD while preserving H and W.

    This replaces the previous full-spatial Tucker call, which was expensive and
    did not improve the spatial domain because the spatial ranks were H and W.
    """
    x = ensure_3d_image(image).astype(np.float32)
    h, w, c = x.shape
    if method == "none" or c <= n_components:
        return normalize_channels(x)
    if method != "pca":
        raise ValueError("method must be 'pca' or 'none'")

    mask = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    flat = x.reshape(-1, c)
    train = flat[mask.ravel()]
    train = train[np.all(np.isfinite(train), axis=1)]
    if len(train) < max(2, n_components):
        return normalize_channels(x[..., :n_components])

    mean = train.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(flat - mean, nan=0.0, posinf=0.0, neginf=0.0)
    _, _, vt = np.linalg.svd(np.nan_to_num(train - mean), full_matrices=False)
    comps = vt[: int(n_components)].T
    projected = centered @ comps
    return normalize_channels(projected.reshape(h, w, int(n_components)))


def postprocess_crown_labels(
    labels: np.ndarray,
    *,
    min_area: int = 20,
    keep_largest_component: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """Clean crown labels by removing tiny regions and optional islands."""
    lab = np.asarray(labels, dtype=np.int32).copy()
    out = np.zeros_like(lab)
    next_label = 1
    for label_id in sorted(int(v) for v in np.unique(lab) if v > 0):
        mask = lab == label_id
        if fill_holes:
            mask = ndi.binary_fill_holes(mask)
        cc, n_cc = ndi.label(mask)
        if n_cc == 0:
            continue
        comps: List[np.ndarray] = []
        for comp_id in range(1, n_cc + 1):
            comp = cc == comp_id
            area = int(comp.sum())
            if area >= int(min_area):
                comps.append(comp)
        if not comps:
            continue
        if keep_largest_component:
            comp = max(comps, key=lambda m: int(m.sum()))
            out[comp] = next_label
            next_label += 1
        else:
            for comp in comps:
                out[comp] = next_label
            next_label += 1
    return out.astype(np.int32, copy=False)


def labels_to_bounding_boxes(labels: np.ndarray, min_area: int = 1) -> List[BBox]:
    """Convert positive label components to [y1, x1, y2, x2] boxes."""
    lab = np.asarray(labels, dtype=np.int32)
    boxes: List[BBox] = []
    for label_id in sorted(int(v) for v in np.unique(lab) if v > 0):
        rows, cols = np.where(lab == label_id)
        if len(rows) < int(min_area):
            continue
        boxes.append([int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())])
    return boxes


# ---------------------------------------------------------------------------
# Visualization and output
# ---------------------------------------------------------------------------


def save_label_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    seed_points_rc: np.ndarray | None,
    output_path: str | Path,
    *,
    alpha: float = 0.55,
) -> None:
    """Save an RGB image overlaid with crown labels and seed points."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.clip(rgb, 0.0, 1.0))
    ax.imshow(labels, cmap="tab20", alpha=float(alpha), vmin=0)
    if seed_points_rc is not None and len(seed_points_rc):
        pts = np.asarray(seed_points_rc)
        ax.scatter(pts[:, 1], pts[:, 0], s=3, c="k")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def show_final_results(
    bestCand: np.ndarray,
    density_map: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    seed_points_rc: np.ndarray,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Display or save the main diagnostic figure."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(density_map, cmap="gray")
    if len(seed_points_rc):
        axes[0].scatter(seed_points_rc[:, 1], seed_points_rc[:, 0], s=2, c="r")
    axes[0].set_title("Density/peaks + seed samples")
    axes[0].axis("off")

    axes[1].imshow(np.clip(rgb, 0.0, 1.0))
    axes[1].set_title("RGB")
    axes[1].axis("off")

    axes[2].imshow(np.clip(rgb, 0.0, 1.0))
    axes[2].imshow(labels, cmap="tab20", alpha=0.55)
    if len(seed_points_rc):
        axes[2].scatter(seed_points_rc[:, 1], seed_points_rc[:, 0], s=2, c="k")
    axes[2].set_title("Propagated crown regions")
    axes[2].axis("off")

    axes[3].imshow(np.clip(bestCand, 0.0, 1.0), cmap="gray")
    axes[3].set_title("Selected ranking feature")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grown_circles_debug(
    image_rgb: np.ndarray,
    peaks_rc: np.ndarray,
    labels: np.ndarray,
    circle_info: Sequence[dict],
    debug: Mapping[str, object],
    alpha: float = 0.35,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Debug plot for circular supports grown from crown peaks."""
    peaks = np.asarray(peaks_rc)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(np.clip(image_rgb, 0.0, 1.0))
    if len(peaks):
        axes[0].scatter(peaks[:, 1], peaks[:, 0], s=15, c="red")
    axes[0].set_title("RGB + peaks")
    axes[0].axis("off")

    axes[1].imshow(debug["vegetation_mask"], cmap="gray")
    axes[1].set_title("Vegetation mask")
    axes[1].axis("off")

    axes[2].imshow(np.clip(image_rgb, 0.0, 1.0))
    axes[2].imshow(labels, cmap="tab20", alpha=alpha)
    axes[2].set_title("Final grown circles")
    axes[2].axis("off")

    axes[3].imshow(np.clip(image_rgb, 0.0, 1.0))
    for info in circle_info:
        r, c = info["center_rc"]
        radius = info["radius"]
        axes[3].add_patch(plt.Circle((c, r), radius, fill=False, linewidth=1.5))
        axes[3].text(c, r, f"{info['circle_id']}\nr={radius}\n{info['stop_reason']}", fontsize=7, ha="center", va="center")
    axes[3].set_title("Radius + stop reason")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """Original FamNet visualization utility retained for compatibility."""
    pred_cnt = output.sum().item() if torch.is_tensor(output) else float(np.asarray(output).sum())
    boxes = boxes.squeeze(0) if torch.is_tensor(boxes) else np.asarray(boxes)
    boxes2 = []
    for i in range(boxes.shape[0]):
        y1, x1, y2, x2 = [int(boxes[i, j].item() if torch.is_tensor(boxes) else boxes[i, j]) for j in (1, 2, 3, 4)]
        roi_cnt = output[0, 0, y1:y2, x1:x2].sum().item() if torch.is_tensor(output) else float(output[y1:y2, x1:x2].sum())
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_) if torch.is_tensor(input_) else input_)
    output_img = format_for_plotting(output)
    if torch.is_tensor(img1):
        img1 = img1.numpy()
    if torch.is_tensor(output_img):
        output_img = output_img.numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(np.clip(img1, 0.0, 1.0))
    for y1, x1, y2, x2, _ in boxes2:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="y", facecolor="none"))
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="k", linestyle="--", facecolor="none"))
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c="red", edgecolors="blue")
        ax.set_title(f"Input image, gt count: {dots.shape[0]}")
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title(f"Overlaid result, predicted count: {pred_cnt:.2f}")
    gray = 0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
    ax.imshow(gray, cmap="gray")
    ax.imshow(output_img, cmap=plt.cm.viridis, alpha=0.5)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title(f"Density map, predicted count: {pred_cnt:.2f}")
    ax.imshow(output_img)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title(f"Density map, predicted count: {pred_cnt:.2f}")
    ret_fig = ax.imshow(output_img)
    for y1, x1, y2, x2, roi_cnt in boxes2:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="y", facecolor="none"))
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="k", linestyle="--", facecolor="none"))
        ax.text(x1, y1, f"{roi_cnt:.2f}", backgroundcolor="y")
    fig.colorbar(ret_fig, ax=ax)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
