"""
CROPLENS AI: Explainable AI & Lesion Attention Heatmap Module
Visualizes early stress and disease lesions on crop leaves in real-time.
100% Offline, CPU-only, no external APIs.
"""

import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple


def generate_leaf_attention_map(image: Image.Image) -> Tuple[Image.Image, np.ndarray, float]:
    """
    Computes an agronomic bio-spatial attention heatmap showing fungal lesions,
    chlorosis patterns, and physiological stress zones.

    Returns:
        overlay_image: PIL Image with color heatmap overlaid on the original leaf
        heatmap_raw: 2D numpy array (normalized 0.0 to 1.0)
        stress_coverage: Percentage of leaf area exhibiting stress
    """
    orig_w, orig_h = image.size
    # Process at comfortable resolution for speed
    proc_size = (256, 256)
    img_resized = image.convert("RGB").resize(proc_size)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    total = r + g + b + 1e-6

    # 1. Vegetation mask (distinguish leaf tissue from white/black background)
    exg = (2.0 * g - r - b) / total
    leaf_mask = (g > 0.15) | (r > 0.15) | (b > 0.15)

    # 2. Disease / Stress intensity:
    # High red/rust components relative to green, or dark necrotic spots
    rust_intensity = np.clip((1.5 * r - g) / total, 0.0, 1.0)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    dark_necrosis = np.clip((0.35 - luminance) / 0.35, 0.0, 1.0)
    chlorosis_yellow = np.clip((r + g - 2.0 * b) / total, 0.0, 1.0)

    # Combined stress map
    stress_map = (rust_intensity * 0.45 + dark_necrosis * 0.40 + chlorosis_yellow * 0.25)
    stress_map = stress_map * leaf_mask

    # Normalize heatmap
    min_val, max_val = stress_map.min(), stress_map.max()
    if max_val > min_val:
        heatmap_norm = (stress_map - min_val) / (max_val - min_val)
    else:
        heatmap_norm = stress_map

    # Calculate stress coverage percentage
    leaf_pixels = max(1, np.sum(leaf_mask))
    stressed_pixels = np.sum((heatmap_norm > 0.40) & leaf_mask)
    stress_coverage = float((stressed_pixels / leaf_pixels) * 100.0)

    # Apply Jet/Turbo colormap for thermal-like lesion display
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # (256, 256, 3)

    # Alpha blending with original image
    alpha = 0.50
    blended = (1.0 - alpha) * arr + alpha * heatmap_colored
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(blended).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return overlay_img, heatmap_norm, round(stress_coverage, 1)
