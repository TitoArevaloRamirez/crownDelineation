
import numpy as np
from PIL import Image
import cv2

import cv2
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import visualize_output_and_save, select_exemplar_rois
from PIL import Image
import os
import torch
import argparse
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss, format_for_plotting
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler

import seaborn as sns; sns.set()


import graphlearning as gl


from skimage.feature import peak_local_max



from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


import matplotlib.pyplot as plt


from skimage.metrics import structural_similarity as ssim


from sklearn.neighbors import kneighbors_graph

from scipy import sparse

from scipy.optimize import minimize

from scipy.special import softmax

import math


parser = argparse.ArgumentParser(description="Few Shot Counting Demo code")
parser.add_argument("-o", "--output-dir", type=str, default=".", help="/Path/to/output/image/file")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")

parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")

args = parser.parse_args()



use_gpu = False
print("===> Using CPU mode.")

resnet50_conv = Resnet50FPN()
regressor = CountRegressor(6, pool='mean')

if use_gpu:
    resnet50_conv.cuda()
    regressor.cuda()
    regressor.load_state_dict(torch.load(args.model_path))
else:
    regressor.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

resnet50_conv.eval()
regressor.eval()



def readData():
    root = "/Users/titoarevalo-ramirez/Data/Talca2025/Registered/Darwin/"
    
    b1 = np.array(Image.open(root + "b_2020_11_21_1.tif"))
    b2 = np.array(Image.open(root + "b_2020_11_21_2.tif"))
    b3 = np.array(Image.open(root + "b_2020_11_22_1.tif"))
                                                          
    g1 = np.array(Image.open(root + "g_2020_11_21_1.tif"))
    g2 = np.array(Image.open(root + "g_2020_11_21_2.tif"))
    g3 = np.array(Image.open(root + "g_2020_11_22_1.tif"))
                                                          
    r1 = np.array(Image.open(root + "r_2020_11_21_1.tif"))
    r2 = np.array(Image.open(root + "r_2020_11_21_2.tif"))
    r3 = np.array(Image.open(root + "r_2020_11_22_1.tif"))

    rEd1 = np.array(Image.open(root + "rEd_2020_11_21_1.tif"))
    rEd2 = np.array(Image.open(root + "rEd_2020_11_21_2.tif"))
    rEd3 = np.array(Image.open(root + "rEd_2020_11_22_1.tif"))
                                                              
    nir1 = np.array(Image.open(root + "nir_2020_11_21_1.tif"))
    nir2 = np.array(Image.open(root + "nir_2020_11_21_2.tif"))
    nir3 = np.array(Image.open(root + "nir_2020_11_22_1.tif"))  

    [n, m] = np.shape(b1)
    b1_r = b1.reshape((1,n, m))
    b2_r = b2.reshape((1,n, m))
    b3_r = b3.reshape((1,n, m))

    g1_r = g1.reshape((1,n, m))
    g2_r = g2.reshape((1,n, m))
    g3_r = g3.reshape((1,n, m))

    r1_r = r1.reshape((1,n, m))
    r2_r = r2.reshape((1,n, m))
    r3_r = r3.reshape((1,n, m))

    rEd1_r = rEd1.reshape((1,n, m))
    rEd2_r = rEd2.reshape((1,n, m))
    rEd3_r = rEd3.reshape((1,n, m))

    nir1_r = nir1.reshape((1,n, m))
    nir2_r = nir2.reshape((1,n, m))
    nir3_r = nir3.reshape((1,n, m))

    b_t = np.concatenate((b1_r, b2_r, b3_r), axis=0)
    g_t = np.concatenate((g1_r, g2_r, g3_r), axis=0)
    r_t = np.concatenate((r1_r, r2_r, r3_r), axis=0)
    rEd_t = np.concatenate((rEd1_r, rEd2_r, rEd3_r), axis=0)
    nir_t = np.concatenate((nir1_r, nir2_r, nir3_r), axis=0)

    data_np = np.stack((b_t, g_t, r_t, rEd_t, nir_t), axis=3)
    return data_np


def normalizeTucker(data_tucker):
    T, H, W, B = data_tucker.shape
    Xn = data_tucker.copy().astype(np.float64)
    
    for b in range(B):
        for t in range(T):
            channel = Xn[t,:, :, b]
            Xn[t, :, :, b] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    return Xn


def compute_vegetation_indices(Xn, eps=1e-8, normalize=True, scale_01=False):
    """
    Compute vegetation indices and normalize them so min = 0.

    Parameters
    ----------
    Xn : np.ndarray
        Shape: (day, width, height, 5)
        Channels: [blue, green, red, red_edge, nir]

    eps : float
        Small constant to avoid division by zero.

    normalize : bool
        If True, shifts each index so its minimum is 0.

    scale_01 : bool
        If True, also scales each index to [0,1].

    Returns
    -------
    dict[str, np.ndarray]
        Each index has shape (day, width, height)
    """

    Xn = Xn.astype(np.float32, copy=False)

    B = Xn[..., 0]
    G = Xn[..., 1]
    R = Xn[..., 2]
    RE = Xn[..., 3]
    NIR = Xn[..., 4]

    def safe_div(a, b):
        return a / (b + eps)

    indices = {}
#EVI, RVI, MTCI, MCARI, CI_red_edge, TCI, MSR, SIPI
    # -------------------------
    # Structural
    # -------------------------
    indices["NDVI"] = safe_div(NIR - R, NIR + R)
    #indices["EVI"] = 2.5 * safe_div(NIR - R, NIR + 6*R - 7.5*B + 1)
    indices["SAVI"] = 1.5 * safe_div(NIR - R, NIR + R + 0.5)
    indices["OSAVI"] = 1.16 * safe_div(NIR - R, NIR + R + 0.16)

    msavi_term = (2*NIR + 1)**2 - 8*(NIR - R)
    msavi_term = np.maximum(msavi_term, 0)
    indices["MSAVI"] = (2*NIR + 1 - np.sqrt(msavi_term)) / 2

    indices["DVI"] = NIR - R
    #indices["RVI"] = safe_div(NIR, R)

    # -------------------------
    # Chlorophyll
    # -------------------------
    indices["NDRE"] = safe_div(NIR - RE, NIR + RE)
    #indices["CI_red_edge"] = safe_div(NIR, RE) - 1
    indices["GNDVI"] = safe_div(NIR - G, NIR + G)
    #indices["MTCI"] = safe_div(NIR - RE, RE - R)
    #indices["MCARI"] = ((RE - R) - 0.2*(RE - G)) * safe_div(RE, R)
    #indices["TCI"] = 1.2*(RE - G) - 1.5*(R - G)*np.sqrt(safe_div(RE, R))

    # -------------------------
    # Nitrogen
    # -------------------------
    nir_r = safe_div(NIR, R)
    #indices["MSR"] = safe_div(nir_r - 1, np.sqrt(nir_r + 1))
    indices["NRI"] = safe_div(R, R + G + B)
    indices["NDVI_RE"] = safe_div(NIR - RE, NIR + RE)

    # -------------------------
    # RGB-based
    # -------------------------
    indices["VARI"] = safe_div(G - R, G + R - B)
    indices["PPR"] = safe_div(G - B, G + B)
    #indices["SIPI"] = safe_div(NIR - B, NIR - R)
    indices["ARVI"] = safe_div(NIR - (2*R - B), NIR + (2*R - B))

    # -------------------------
    # NORMALIZATION
    # -------------------------
    if normalize:
        for k, v in indices.items():
            v_min = np.nanmin(v)
            v = v - v_min  # shift so min = 0

            if scale_01:
                v_max = np.nanmax(v)
                v = v / (v_max + eps)

            indices[k] = v

    return indices

def plot_vegetation_indices(indices_dict, day=0, cmap="RdYlGn", figsize=(15, 10)):
    """
    Plot vegetation indices in subplots.

    Parameters
    ----------
    indices_dict : dict[str, np.ndarray]
        Dictionary of vegetation indices.
        Each value must have shape (day, width, height)

    day : int
        Which temporal slice to plot.

    cmap : str
        Colormap for visualization.

    figsize : tuple
        Size of the matplotlib figure.
    """

    names = list(indices_dict.keys())
    n = len(names)

    # Compute grid size (square-ish)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(names):
        ax = axes[i]

        img = indices_dict[name][day]

        im = ax.imshow(img, cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

        # Add colorbar per subplot (optional but useful)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    #plt.tight_layout()



def build_candidate_features(Xn, indices_dict, day=None):
    """
    Build dict of raw channels + vegetation indices.

    If day is not None, returns 2D images for that day.
    Otherwise returns 3D arrays (day, width, height).
    """
    if day is None:
        candidates = {
            "Blue": Xn[..., 0],
            "Green": Xn[..., 1],
            "Red": Xn[..., 2],
            "RedEdge": Xn[..., 3],
            "NIR": Xn[..., 4],
        }
        candidates.update(indices_dict)
    else:
        candidates = {
            "Blue": Xn[day,:,:, 0],
            "Green": Xn[day,:,:, 1],
            "Red": Xn[day,:,:, 2],
            "RedEdge": Xn[day,:,:, 3],
            "NIR": Xn[day,:,:, 4],
        }
        for k, v in indices_dict.items():
            candidates[k] = v[day]

    return candidates

def maskBrightSpots(candidates, laplacian_ksize=3, dilate_ksize=5):
    """
    Generate one bright-spot mask per candidate feature.

    Parameters
    ----------
    candidates : dict[str, np.ndarray]
        Dict of 2D feature images.
    laplacian_ksize : int
        Kernel size for Laplacian.
    dilate_ksize : int
        Kernel size for dilation.

    Returns
    -------
    masks : dict[str, np.ndarray]
        Dict of binary masks (uint8, values 0 or 255), one per candidate.
    """
    masks = {}

    for name, img in candidates.items():
        img = np.asarray(img, dtype=np.float32)

        finite = np.isfinite(img)
        if not np.any(finite):
            masks[name] = np.zeros(img.shape, dtype=np.uint8)
            continue

        vals = img[finite]
        vmin = np.min(vals)
        vmax = np.max(vals)

        if vmax - vmin < 1e-8:
            gray_image = np.zeros(img.shape, dtype=np.uint8)
        else:
            norm_img = (img - vmin) / (vmax - vmin)
            norm_img[~finite] = 0.0
            gray_image = np.asarray(255.0 * norm_img, dtype=np.uint8)

        # Extract edges
        lap = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=laplacian_ksize)
        lap = np.abs(lap)
        lap = np.clip(lap, 0, 255).astype(np.uint8)

        # Fill/expand edge regions
        kernel = np.ones((dilate_ksize, dilate_ksize), dtype=np.uint8)

        #dilated = cv2.morphologyEx(lap, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(lap, kernel, iterations=1)

        # Threshold
        _, thresh = cv2.threshold(
            dilated,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        masks[name] = thresh

    return masks

def compute_feature_metrics(feature_img, mask, eps=1e-8):
    """
    Compute separability and blob metrics for one feature image and one mask.

    Parameters
    ----------
    feature_img : np.ndarray
        2D array, one channel or one vegetation index.
    mask : np.ndarray
        Binary mask, same HxW shape as feature_img. Nonzero = bright spot.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    metrics : dict
    """
    feature_img = np.asarray(feature_img, dtype=np.float32)
    mask = np.asarray(mask) > 0

    if feature_img.shape != mask.shape:
        raise ValueError(f"Shape mismatch: feature_img {feature_img.shape}, mask {mask.shape}")

    inside = feature_img[mask]
    outside = feature_img[~mask]

    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    blob_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)

    n_blobs = len(blob_areas)
    bright_fraction = float(mask.mean())
    mean_blob_area = float(blob_areas.mean()) if n_blobs > 0 else 0.0
    max_blob_area = float(blob_areas.max()) if n_blobs > 0 else 0.0

    if inside.size == 0 or outside.size == 0:
        return {
            "mu_in": 0.0,
            "mu_out": 0.0,
            "std_in": 0.0,
            "std_out": 0.0,
            "contrast_ratio": 0.0,
            "effect_size": 0.0,
            "fisher_score": 0.0,
            "bright_fraction": bright_fraction,
            "n_blobs": n_blobs,
            "mean_blob_area": mean_blob_area,
            "max_blob_area": max_blob_area,
        }

    mu_in = float(np.mean(inside))
    mu_out = float(np.mean(outside))
    std_in = float(np.std(inside))
    std_out = float(np.std(outside))

    contrast_ratio = (mu_in - mu_out) / (abs(mu_out) + eps)
    pooled_std = np.sqrt((std_in**2 + std_out**2) / 2.0)
    effect_size = (mu_in - mu_out) / (pooled_std + eps)
    fisher_score = ((mu_in - mu_out) ** 2) / (std_in**2 + std_out**2 + eps)

    return {
        "mu_in": mu_in,
        "mu_out": mu_out,
        "std_in": std_in,
        "std_out": std_out,
        "contrast_ratio": float(contrast_ratio),
        "effect_size": float(effect_size),
        "fisher_score": float(fisher_score),
        "bright_fraction": bright_fraction,
        "n_blobs": n_blobs,
        "mean_blob_area": mean_blob_area,
        "max_blob_area": max_blob_area,
    }


def rank_candidates_with_mask(candidates, masks=None, sort_by="fisher_score"):
    """
    Compute metrics for every candidate using its own mask.

    Parameters
    ----------
    candidates : dict[str, np.ndarray]
        Dict of 2D feature images.
    masks : dict[str, np.ndarray] or None
        Dict of masks keyed by candidate name.
        If None, masks are generated with maskBrightSpots(candidates).
    sort_by : str
        Metric to rank by.

    Returns
    -------
    results : list[dict]
        Ranked list of metrics, highest first.
    """
    if masks is None:
        masks = maskBrightSpots(candidates)

    results = []

    for name, feature_img in candidates.items():
        if name not in masks:
            continue

        metrics = compute_feature_metrics(feature_img, masks[name])
        metrics["name"] = name
        results.append(metrics)

    results.sort(key=lambda x: x[sort_by], reverse=True)
    return results

def plot_top_candidates_with_overlay(candidates, masks, results, k=5, cmap="viridis"):
    """
    Plot top-k candidates with mask overlay.

    Parameters
    ----------
    candidates : dict[str, np.ndarray]
        Feature images (2D).

    masks : dict[str, np.ndarray]
        Binary masks per candidate.

    results : list[dict]
        Output from rank_candidates_with_mask().

    k : int
        Number of top candidates to plot.

    cmap : str
        Colormap for feature images.
    """

    k = min(k, len(results))
    top = results[:k]

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 5))

    if k == 1:
        axes = [axes]

    for i, r in enumerate(top):
        name = r["name"]

        img = candidates[name]
        mask = masks[name] > 0

        ax = axes[i]

        # Normalize image for display
        img_disp = img.astype(np.float32)
        finite = np.isfinite(img_disp)
        if np.any(finite):
            vmin = np.percentile(img_disp[finite], 2)
            vmax = np.percentile(img_disp[finite], 98)
            img_disp = np.clip(img_disp, vmin, vmax)
            img_disp = (img_disp - vmin) / (vmax - vmin + 1e-8)
        else:
            img_disp = np.zeros_like(img_disp)

        # Show base image
        ax.imshow(img_disp, cmap=cmap)

        # Create red overlay for mask
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0  # red channel
        overlay[..., 3] = mask.astype(np.float32) * 0.4  # alpha

        ax.imshow(overlay)

        ax.set_title(
            f"{name}\nF={r['fisher_score']:.2f}, blobs={r['n_blobs']}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def detect_big_round_blobs(
    mask,
    min_area=100,
    min_circularity=0.6,
    min_solidity=0.85,
    morph_open_ksize=3,
    morph_close_ksize=5,
):
    """
    Detect big and round white blobs from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary image. White blobs should be > 0.
    min_area : int
        Minimum blob area in pixels.
    min_circularity : float
        Minimum circularity in [0, 1].
    min_solidity : float
        Minimum solidity in [0, 1].
    morph_open_ksize : int
        Kernel size for opening to remove noise.
    morph_close_ksize : int
        Kernel size for closing to fill small gaps.

    Returns
    -------
    selected_mask : np.ndarray
        Binary mask with only selected blobs.
    blob_info : list[dict]
        Metrics for each selected blob.
    contours_selected : list
        Selected contours.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    # Ensure binary uint8
    bw = (mask > 0).astype(np.uint8) * 255

    # Clean mask
    if morph_open_ksize > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open)

    if morph_close_ksize > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    # Find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    selected_mask = np.zeros_like(bw)
    contours_selected = []
    blob_info = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        if circularity < min_circularity:
            continue
        if solidity < min_solidity:
            continue

        # Optional ellipse-based roundness if enough points exist
        roundness = np.nan
        if len(cnt) >= 5:
            (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
            if MA > 0 and ma > 0:
                a = max(ma, MA) / 2.0
                b = min(ma, MA) / 2.0
                roundness = b / a  # 1.0 is perfect circle

        cv2.drawContours(selected_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        contours_selected.append(cnt)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = np.nan, np.nan

        blob_info.append(
            {
                "area": float(area),
                "perimeter": float(perimeter),
                "circularity": float(circularity),
                "solidity": float(solidity),
                "roundness": float(roundness) if np.isfinite(roundness) else np.nan,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
            }
        )

    return selected_mask, blob_info, contours_selected

def mask_from_selected_contours(shape, contours, border_margin=1):
    """
    Create binary mask from selected contours, excluding blobs touching image borders.

    Parameters
    ----------
    shape : tuple
        Shape of the output mask (H, W)
    contours : list
        List of contours
    border_margin : int
        Margin from border to consider as touching (default=1 pixel)

    Returns
    -------
    mask : np.ndarray
        Binary mask with filtered blobs
    filtered_contours : list
        Contours that were kept
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)

    filtered_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Check if contour touches border
        touches_border = (
            x <= border_margin or
            y <= border_margin or
            (x + w) >= (W - border_margin) or
            (y + h) >= (H - border_margin)
        )

        if touches_border:
            continue

        filtered_contours.append(cnt)

    if len(filtered_contours) > 0:
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    return mask, filtered_contours

def get_top_k_blobs(blob_info, contours, k=3, mode="area_circularity"):
    """
    Select top-k blobs based on a scoring criterion.

    Parameters
    ----------
    blob_info : list[dict]
        Output from detect_big_round_blobs()
    contours : list
        Corresponding contours
    k : int
        Number of blobs to return
    mode : str
        Scoring method:
        - "area"
        - "circularity"
        - "area_circularity" (recommended)

    Returns
    -------
    selected_info : list[dict]
    selected_contours : list
    """

    if len(blob_info) == 0:
        return [], []

    scores = []

    for b in blob_info:
        if mode == "area":
            s = b["area"]
        elif mode == "circularity":
            s = b["circularity"]
        else:
            s = b["area"] * b["circularity"]

        scores.append(s)

    scores = np.array(scores)
    idx_sorted = np.argsort(scores)[::-1]  # descending

    top_idx = idx_sorted[:k]

    selected_info = [blob_info[i] for i in top_idx]
    selected_contours = [contours[i] for i in top_idx]

    return selected_info, selected_contours

def get_blob_bounding_boxes(contours):
    boxes = []

    min_area = np.finfo(np.float64).max

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        x1 = x
        y1 = y
        x2 = x1 + w - 1
        y2 = y1 + h - 1

        area = int(w * h)
        if area <= min_area:
            min_area = area

        boxes.append([y1, x1, y2, x2])

    return boxes, min_area

def sample_points_in_circle(
    center,
    area,
    num_points=1,
    image_shape=None,   # (H, W)
    mode="resample",    # "resample" or "clip"
    seed=None,
):
    """
    Sample random (x, y) points uniformly inside a circle,
    optionally constrained to image bounds.

    Parameters
    ----------
    center : tuple
        (cx, cy)
    area : float
        Circle area
    num_points : int
        Number of points to sample
    image_shape : tuple or None
        (H, W). If provided, points are constrained within image.
    mode : str
        "resample" → discard out-of-bounds and resample (recommended)
        "clip" → clip coordinates to valid range
    seed : int or None
        Random seed

    Returns
    -------
    points : np.ndarray
        Shape (num_points, 2)
    """
    rng = np.random.default_rng(seed)

    cx, cy = center
    R = np.sqrt(area / np.pi)

    def sample(n):
        u = rng.random(n)
        v = rng.random(n)
        r = R * np.sqrt(u)
        theta = 2 * np.pi * v

        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)

        return np.column_stack((x, y))

    # No bounds → simple case
    if image_shape is None:
        return sample(num_points)

    H, W = image_shape

    if mode == "clip":
        pts = sample(num_points)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        return pts

    elif mode == "resample":
        pts_list = []
        remaining = num_points

        while remaining > 0:
            pts = sample(remaining * 2)  # oversample for efficiency

            valid = (
                (pts[:, 0] >= 0) & (pts[:, 0] < W) &
                (pts[:, 1] >= 0) & (pts[:, 1] < H)
            )

            pts_valid = pts[valid]

            if len(pts_valid) > 0:
                take = min(len(pts_valid), remaining)
                pts_list.append(pts_valid[:take])
                remaining -= take

        return np.vstack(pts_list)

    else:
        raise ValueError("mode must be 'resample' or 'clip'")


def detect_strong_peaks(image, min_distance=5, threshold_rel=None):
    """
    Detect local maxima greater than the image median.

    Parameters
    ----------
    image : np.ndarray
        2D image
    min_distance : int
        Minimum distance between peaks
    threshold_rel : float or None
        Optional relative threshold (0–1)

    Returns
    -------
    coords : np.ndarray
        Peak coordinates (N, 2)
    """

    image = np.asarray(image, dtype=np.float32)

    # Compute median threshold
   # med = np.median(image)
    med = np.percentile(image, 80)

    # Option 1: use threshold_abs
    coords = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=med,
        threshold_rel=threshold_rel,
        exclude_border=True
    )

    image[image<med] = 0

    return coords, image



def main():
    data = readData();
    data_tucker = data[:,1024:1544,1024:1544,:]
    Xn = normalizeTucker(data_tucker) #day:w:h:channel
    indices = compute_vegetation_indices(Xn)
    
    #plot_vegetation_indices(indices, day=0)
    #plot_vegetation_indices(indices, day=1)
    #plot_vegetation_indices(indices, day=2)
    #plt.show()

    day = 0
    candidates_0 = build_candidate_features(Xn, indices, day)
    day = 1
    candidates = build_candidate_features(Xn, indices, day)

    masks = maskBrightSpots(candidates)
    results = rank_candidates_with_mask(candidates, masks=masks, sort_by="contrast_ratio")
    print(results[0]['name'])

    bgr_0 = np.stack((candidates_0["Blue"], candidates_0["Green"],candidates_0["Red"]), 2)
    bgr_1 = np.stack((candidates["Blue"], candidates["Green"],candidates["Red"]), 2)
    img = candidates[results[0]['name']]
    img_3channel = np.stack((img, img, img), 2)

    image = Image.fromarray(np.uint8(img_3channel*255))

    mask = masks[results[0]['name']]

    selected_mask, blob_info, contours = detect_big_round_blobs(mask)
    
    top_info, top_contours = get_top_k_blobs(blob_info, contours, k=5)
    
    top_mask, filtered_contours = mask_from_selected_contours(mask.shape, top_contours)

    boxes, min_area = get_blob_bounding_boxes(filtered_contours)

    if False:
        boxes, min_area = get_blob_bounding_boxes(filtered_contours)
        rects1 = list()
        for rect in boxes:
            y1, x1, y2, x2 = rect
            rects1.append([y1, x1, y2, x2])
        
        sample = {'image': image, 'lines_boxes': rects1}

        sample = Transform(sample)
        image, boxes = sample['image'], sample['boxes']

        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        
        features.required_grad = True
        #adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor = regressor
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)

        pbar = tqdm(range(args.gradient_steps))
        for step in pbar:
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes, use_gpu=use_gpu)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=use_gpu)
            Loss = lCount + lPerturbation
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()

            pbar.set_description('Adaptation step: {:<3}, loss: {}, predicted-count: {:6.1f}'.format(step, Loss.item(), output.sum().item()))

        features.required_grad = False
        output = adapted_regressor(features)
        print(output)

        #np.save("./countr_output.npy", output)
        torch.save(output, 'tensor_output.pt') #

    output = torch.load('tensor_output.pt') #
    density_map = np.asarray(format_for_plotting(output))

    print(np.min(density_map))
    print(np.max(density_map))

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(bgr_0)
    #plt.imshow(density_map, alpha=0.5)
    #plt.subplot(1,2,2)
    #plt.imshow(bgr_1)
    #plt.imshow(density_map, alpha=0.5)
    #plt.show()

    density_map_gauss = np.copy(density_map)
    density_map_gauss = ndi.gaussian_filter(density_map_gauss, sigma=3)
    
    #coordinates = peak_local_max(density_map_gauss, min_distance=15)

    coordinates, density_map_mask = detect_strong_peaks(density_map, min_distance=15)
    
    print("Coordinates of local maxima (row, col):")
    print(coordinates)
    
    # 4. (Optional) Visualize the result
    n_c_ex = 40
    n_c = coordinates.shape[0]
    n_ex = n_c*n_c_ex           # Total number of labeled samples. 


    names = list(candidates_0.keys())
    n = len(names)

    data_matrix = None
    for i, name in enumerate(names):
        img_data = np.asarray(candidates_0[name]).reshape(img.shape[0], img.shape[1],1 )

        if data_matrix is None:
            data_matrix = img_data
        else:
            data_matrix = np.concatenate((data_matrix, img_data), axis=2)

    #distance = ndi.distance_transform_edt(density_map)

    #data_matrix = np.concatenate((density_map.reshape((520,520, 1)), distance.reshape((520,520, 1)), np.asanyarray(data_matrix)), axis=2)
    #data_matrix = np.concatenate((distance.reshape((520,520, 1)), np.asanyarray(data_matrix)), axis=2)
    data_matrix = np.concatenate((density_map.reshape((520,520, 1)), np.asanyarray(data_matrix)), axis=2)
    print(data_matrix.shape)

    x_coords, y_coords = np.meshgrid(np.arange(520), np.arange(520))
    
    # Flatten the coordinate arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()


    #bgr = np.copy(data_matrix[:,:,[1,2,3]])
    #data_flat = data_matrix.reshape(-1,17)
    #print(data_flat.shape)

    n_clusters = np.int32(np.rint(output.sum().item()))

    labels_clusters = None;
    data_pts = None
    print(n_c)
    print(n_ex)
    labels_img = np.zeros((520,520,1))

    n_labels = 0
    for i in range(0, n_c):
        pts = sample_points_in_circle(center=coordinates[i,:], area=min_area,
        num_points= n_c_ex,
        image_shape=(520-1, 520-1), #W, H
        mode="resample")


        pts = np.int32(np.rint(pts))

        for j in range(0, pts.shape[0]):
            feat_data = np.asarray(data_matrix[pts[j, 0], pts[j, 1], :]).reshape((1, data_matrix.shape[2]))
            feat_data = np.concatenate((np.array([pts[j, 0], pts[j, 1]]).reshape((1,2)), feat_data), axis=1)

            labels_img[pts[j, 0], pts[j, 1], 0] = n_labels

            if data_pts is None:
                data_pts =  feat_data
            else:
                data_pts = np.concatenate((data_pts,  feat_data ), axis=0)


        labels = np.ones((n_c_ex, 1))*(i)
        #labels = np.ones((n_c_ex, 1))*n_labels#*(i)
        n_labels = n_labels +1;
        if n_labels == 5:
            n_labels = 0

        if labels_clusters is None:
            labels_clusters = labels 
        else:
            labels_clusters = np.concatenate((labels_clusters, labels), axis= 0)

    print(data_pts.shape)
    print(labels_clusters.shape)

    data_flat = data_matrix.reshape(-1,18)
    print(data_pts)
    print(data_pts.shape)

    # Create a StandardScaler object
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(data_flat[:,0:5])
    x_scaled = scaler.transform(data_pts[:,2:7])
    #x_scaled = data_pts[:, 2:6]
    labels_clusters = np.int16(labels_clusters).reshape(-1)

    k = 10 
    metric = 'vae' 
    
    W = gl.weightmatrix.knn(x_scaled, k, metric=metric)
    D = gl.weightmatrix.knn(x_scaled, k, metric=metric, kernel='distance')

    ntrain = x_scaled.shape[0]

    print(ntrain)
    print(X_scaled.shape[0])

    X = np.concatenate((x_scaled, X_scaled), axis=0)

    print(X.shape[0])

    W_real = gl.weightmatrix.knn(X, k, metric=metric)


    #class_priors = gl.utils.class_priors(labels_clusters)

    ##
    #train_ind = gl.trainsets.generate(labels_clusters, rate=30)
    #train_labels = labels_clusters[train_ind]

    train_ind = range(0, ntrain) 
    train_labels = labels_clusters

    ##
    #model1 = gl.ssl.laplace(W)
    #model2 = gl.ssl.graph_nearest_neighbor(D, class_priors=class_priors)
    #model3 = gl.ssl.laplace(W, reweighting='wnll')
    model4 = gl.ssl.laplace(W_real, reweighting='poisson')
    #model5 = gl.ssl.poisson(W_real, solver='gradient_descent')
    #model6 = gl.ssl.dynamic_label_propagation(W, class_priors=class_priors)
    #model7 = gl.clustering.incres(W, num_clusters=n_c)


    ##pred_labels_incres = model.fit_predict(all_labels=labels_clusters)

    ##accuracy = gl.clustering.clustering_accuracy(pred_labels_incres,labels_clusters)
    ##print('Clustering Accuracy Incres: %.2f%%'%accuracy)

    pred_labels4 = gl.clustering.RP1D(X,100)



    #pred_labels1 = model1.fit_predict(train_ind, train_labels )
    #pred_labels2 = model2.fit_predict(train_ind, train_labels )
    #pred_labels3 = model3.fit_predict(train_ind, train_labels )
    #pred_labels3 = model3.fit_predict(train_ind, train_labels )

    #pred_labels4 = model4.fit_predict(train_ind, train_labels )
    #np.save("./pred_labels4.npy", pred_labels4)
    #pred_labels4 = np.load("./pred_labels4.npy")

    #pred_labels5 = model5.fit_predict(train_ind, train_labels )
    #np.save("./pred_labels5.npy", pred_labels5)
    #pred_labels5 = np.load("./pred_labels5.npy")

    #pred_labels6 = model6.fit_predict(train_ind, train_labels )
    #pred_labels6 = model6.fit_predict(train_ind, train_labels )

    #accuracy1 = gl.ssl.ssl_accuracy(pred_labels1, labels_clusters, train_ind)   
    #accuracy2 = gl.ssl.ssl_accuracy(pred_labels2, labels_clusters, train_ind)   
    #accuracy3 = gl.ssl.ssl_accuracy(pred_labels3, labels_clusters, train_ind)   
    accuracy4 = gl.ssl.ssl_accuracy(pred_labels4[0:ntrain], labels_clusters, train_ind)   
    #accuracy5 = gl.ssl.ssl_accuracy(pred_labels5[0:ntrain], labels_clusters, train_ind)   
    #accuracy6 = gl.ssl.ssl_accuracy(pred_labels6, labels_clusters, train_ind)   
    #print("Accuracy1: %.2f%%"%accuracy1)
    #print("Accuracy2: %.2f%%"%accuracy2)
    #print("Accuracy3: %.2f%%"%accuracy3)
    print("Accuracy4: %.2f%%"%accuracy4)
    #print("Accuracy5: %.2f%%"%accuracy5)
    #print("Accuracy6: %.2f%%"%accuracy6)


    pred_labels5_img = pred_labels4[ntrain:273080]

    print(np.max(pred_labels5_img))

    pred_labels5_img = pred_labels5_img.reshape((520,520))

    #pred_labels5_img[pred_labels5_img!=60] = 0

    print(pred_labels5_img.shape)

    img_reshp = X[ntrain:273080,3].reshape((520, 520))

    plt.figure()
    plt.imshow(bgr_0 )
    plt.imshow(pred_labels5_img, alpha = 0.5)
    #plt.scatter(data_pts[:,1],data_pts[:,0], c=pred_labels4[0:ntrain], s=1)
    plt.show()



    ###model = gl.ssl.poisson_mbo(W, class_priors)
    ###pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels_clusters)
    ###
    ###accuracy = gl.ssl.ssl_accuracy(labels_clusters,pred_labels,train_ind)
    ###print(model.name + ': %.2f%%'%accuracy)


    ###print(labels_clusters)

 
    #plt.figure()
    #plt.subplot(2, 3, 1)
    #plt.scatter(data_pts[:,0],data_pts[:,1], c=pred_labels1, s=1)
    #plt.title("model 1")
    #plt.subplot(2, 3, 2)
    #plt.scatter(data_pts[:,0],data_pts[:,1], c=pred_labels2, s=1)
    #plt.title("model 2")
    #plt.subplot(2, 3, 3)
    #plt.scatter(data_pts[:,0],data_pts[:,1], c=pred_labels3, s=1)
    #plt.title("model 3")
    #plt.subplot(2, 3, 4)
    #plt.scatter(data_pts[:,0],data_pts[:,1], c=pred_labels4, s=1)
    #plt.title("model 4")
    #plt.subplot(2, 3, 5)
    #plt.imshow(bgr_0)
    #plt.scatter(data_pts[:,1],data_pts[:,0], c=pred_labels6, s=1)
    #plt.title("model 5")
    #plt.subplot(2, 3, 6)
    #plt.scatter(data_pts[:,1],data_pts[:,0], c=pred_labels6, s=1)
    ##plt.plot(data_pts[:,1],data_pts[:,0], 'b+')
    #plt.title("model 6")
    ##plt.plot(data_pts[:,1],data_pts[:,0], 'b+')
    #plt.show()


    #pred_all3 = model3.predict(X_scaled)

    #print(pred_all3)
    #print(pred_all3.shape)

    #plt.figure()
    #plt.imshow(bgr_0)
    #plt.imshow(pred_all3.reshape((512,512,1)), alpha=0.5)
    #plt.show()








    #print(density_map.shape)

    #scaler_01 = MinMaxScaler(feature_range=(0, 1))
    #density_map = np.uint8(255*scaler_01.fit_transform(density_map_mask))
    #distance = ndi.distance_transform_edt(density_map)

    #mask = np.zeros(distance.shape, dtype=bool)
    #data_coord = np.uint32(np.copy(data_pts[:,[0, 1]]))
    #mask[tuple(data_coord.T)] = True
    ##mask[tuple(coordinates.T)] = True
    #markers, _ = ndi.label(mask)
    #print(np.max(markers))
    #print(np.shape(markers))
    #print(labels_img.reshape((520, 520)).shape)
    #print(np.max(labels_img))
    #labels = watershed(-distance, markers, mask=img[:,:] )
    #
    #fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    #ax = axes.ravel()
    #
    #ax[0].imshow(markers, cmap=plt.cm.gray)
    #ax[0].set_title('Overlapping objects')
    #ax[1].imshow(-distance, cmap=plt.cm.gray)
    #ax[1].set_title('Distances')
    #ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    #ax[2].plot(data_pts[:,1], data_pts[:,0], 'k+')

    #ax[2].set_title('Separated objects')

    #fig.tight_layout()
    #plt.show()



    #print('===> The predicted count is: {:6.2f}'.format(output.sum().item()))
    #
    #rslt_file = "{}/{}_out.png".format(args.output_dir, "test_countr_crown")
    #visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)
    #print("===> Visualized output is saved to {}".format(rslt_file))


    
    #print(f"Selected {len(top_info)} blobs")
    #for i, b in enumerate(top_info):
    #    print(
    #        f"Blob {i}: area={b['area']:.1f}, "
    #        f"circ={b['circularity']:.3f}"
    #    )

    #plt.figure(figsize=(12, 4))
    #
    #plt.subplot(1, 3, 1)
    #plt.imshow(img, cmap="gray")
    #plt.title("Original mask")
    #plt.axis("off")
    #
    #plt.subplot(1, 3, 2)
    #plt.imshow(selected_mask, cmap="gray")
    #plt.title("All valid blobs")
    #plt.axis("off")
    #
    #plt.subplot(1, 3, 3)
    #plt.imshow(density_map, cmap="gray")
    #plt.title("Top 3 blobs")
    #plt.axis("off")
    #
    #plt.tight_layout()
    #plt.show()

    
    #for r in results[:10]:
    #    print(
    #        f"{r['name']:12s} "
    #        f"Fisher={r['fisher_score']:.4f} "
    #        f"Effect={r['effect_size']:.4f} "
    #        f"Contrast={r['contrast_ratio']:.4f} "
    #        f"Blobs={r['n_blobs']:4d} "
    #        f"Frac={r['bright_fraction']:.4f}"
    #    )

    #plot_top_candidates_with_overlay(candidates, masks, results, k=10)




if __name__ == "__main__":
    main()



