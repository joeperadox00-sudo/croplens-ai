"""
CROPLENS AI - High Quality Sample Leaf Image Generator
Creates realistic plant leaf samples for all 8 diagnostic classes:
  1. Healthy Leaf
  2. Early Blight
  3. Common Rust
  4. Leaf Spot / Late Blight
  5. Nutrient Chlorosis / Water Stress
  6. Aphids / Whiteflies Infestation
  7. Leaf Miners / Armyworms Damage
  8. Spider Mites / Thrips Micro-Stippling
100% Local, zero external dataset downloads required.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_base_leaf(width=380, height=380) -> Image.Image:
    """Draws a natural leaf shape on a neutral background."""
    img = Image.new("RGB", (width, height), color=(240, 245, 240))
    draw = ImageDraw.Draw(img)

    # Leaf silhouette points (elliptical curve tapering at apex and petiole)
    cx, cy = width // 2, height // 2
    points = []
    num_pts = 90
    for i in range(num_pts):
        theta = 2 * math.pi * (i / num_pts)
        r_base = 130
        y_factor = 1.0 + 0.35 * math.cos(theta)
        x_factor = (0.75 + 0.25 * math.sin(theta)) * math.sin(theta)
        r = r_base * y_factor
        x = cx + int(r * math.sin(theta) * 0.9)
        y = cy - int(r * math.cos(theta) * 1.35)
        points.append((x, y))

    # Base vibrant foliage color
    draw.polygon(points, fill=(48, 145, 52))

    # Draw primary leaf vein
    draw.line([(cx, cy - 165), (cx, cy + 165)], fill=(32, 105, 38), width=4)

    # Draw secondary lateral veins
    for dy in range(-120, 130, 28):
        y_pos = cy + dy
        span = int(95 * (1.0 - abs(dy) / 180.0))
        draw.line([(cx, y_pos), (cx - span, y_pos - 35)], fill=(38, 120, 44), width=2)
        draw.line([(cx, y_pos), (cx + span, y_pos - 35)], fill=(38, 120, 44), width=2)

    return img


def create_healthy_sample(output_path: str):
    leaf = generate_base_leaf()
    leaf = leaf.filter(ImageFilter.SMOOTH_MORE)
    leaf.save(output_path, "JPEG", quality=92)


def create_early_blight_sample(output_path: str):
    leaf = generate_base_leaf()
    draw = ImageDraw.Draw(leaf)

    lesion_centers = [(150, 160), (220, 210), (170, 270), (240, 140)]
    for lx, ly in lesion_centers:
        draw.ellipse([lx - 32, ly - 32, lx + 32, ly + 32], fill=(195, 180, 45))
        draw.ellipse([lx - 22, ly - 22, lx + 22, ly + 22], fill=(70, 45, 25))
        draw.ellipse([lx - 15, ly - 15, lx + 15, ly + 15], fill=(110, 75, 40))
        draw.ellipse([lx - 8, ly - 8, lx + 8, ly + 8], fill=(35, 20, 10))

    leaf = leaf.filter(ImageFilter.SMOOTH)
    leaf.save(output_path, "JPEG", quality=92)


def create_common_rust_sample(output_path: str):
    leaf = generate_base_leaf()
    draw = ImageDraw.Draw(leaf)

    np.random.seed(42)
    cx, cy = 190, 190
    for _ in range(85):
        rx = int(np.random.normal(cx, 55))
        ry = int(np.random.normal(cy, 90))
        size = np.random.randint(4, 9)
        color = (
            np.random.randint(185, 225),
            np.random.randint(85, 125),
            np.random.randint(20, 45)
        )
        draw.ellipse([rx - size, ry - size, rx + size, ry + size], fill=color)

    leaf = leaf.filter(ImageFilter.SMOOTH)
    leaf.save(output_path, "JPEG", quality=92)


def create_late_blight_sample(output_path: str):
    leaf = generate_base_leaf()
    draw = ImageDraw.Draw(leaf)

    blotches = [
        [(120, 130), (180, 110), (200, 170), (140, 185)],
        [(190, 220), (265, 210), (250, 290), (175, 275)],
        [(140, 290), (190, 310), (170, 350), (130, 335)]
    ]
    for blotch in blotches:
        draw.polygon(blotch, fill=(42, 32, 22))

    for lx, ly in [(160, 150), (220, 250)]:
        draw.ellipse([lx - 45, ly - 45, lx + 45, ly + 45], outline=(145, 155, 95), width=4)

    leaf = leaf.filter(ImageFilter.SMOOTH)
    leaf.save(output_path, "JPEG", quality=92)


def create_chlorosis_stress_sample(output_path: str):
    leaf = generate_base_leaf()
    arr = np.asarray(leaf, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    is_green = (g > 70) & (g > r)
    arr[is_green, 0] = np.clip(arr[is_green, 0] * 1.55 + 40, 0, 230)
    arr[is_green, 1] = np.clip(arr[is_green, 1] * 1.05, 0, 235)
    arr[is_green, 2] = np.clip(arr[is_green, 2] * 0.50, 0, 80)

    stressed_leaf = Image.fromarray(arr.astype(np.uint8))
    stressed_leaf = stressed_leaf.filter(ImageFilter.SMOOTH)
    stressed_leaf.save(output_path, "JPEG", quality=92)


def create_aphids_sample(output_path: str):
    """Generates leaf with dense clusters of sap-sucking aphids & whiteflies."""
    leaf = generate_base_leaf()
    draw = ImageDraw.Draw(leaf)

    np.random.seed(101)
    # Clusters concentrated along primary and secondary veins
    centers = [(190, 150), (190, 210), (190, 270), (160, 180), (220, 230)]
    for cx, cy in centers:
        # Cluster of 25-35 insects
        for _ in range(30):
            ox = int(np.random.normal(cx, 16))
            oy = int(np.random.normal(cy, 18))
            is_whitefly = np.random.rand() > 0.6
            if is_whitefly:
                color = (245, 245, 235)  # Tiny whitefly
                rad = np.random.randint(2, 4)
            else:
                color = (195, 210, 60)   # Green/yellow aphid nymph
                rad = np.random.randint(3, 5)
            draw.ellipse([ox - rad, oy - rad, ox + rad, oy + rad], fill=color)

    # Some honeydew sticky shiny drops
    for _ in range(25):
        hx = np.random.randint(140, 250)
        hy = np.random.randint(120, 300)
        draw.ellipse([hx - 2, hy - 2, hx + 2, hy + 2], fill=(220, 230, 180))

    leaf.save(output_path, "JPEG", quality=92)


def create_leaf_miner_sample(output_path: str):
    """Generates leaf with winding white serpentine mines and chewed margin notches."""
    leaf = generate_base_leaf()
    draw = ImageDraw.Draw(leaf)

    # Draw continuous serpentine winding trails (mines)
    # Trail 1
    t1_points = [
        (160, 120), (175, 135), (150, 160), (180, 180),
        (165, 210), (145, 230), (160, 260), (190, 280), (210, 270)
    ]
    draw.line(t1_points, fill=(245, 248, 240), width=5)
    draw.line(t1_points, fill=(180, 190, 165), width=2)

    # Trail 2
    t2_points = [
        (220, 130), (205, 155), (235, 175), (215, 205),
        (240, 225), (225, 250), (205, 245)
    ]
    draw.line(t2_points, fill=(245, 248, 240), width=4)
    draw.line(t2_points, fill=(180, 190, 165), width=2)

    # Trail 3
    t3_points = [
        (130, 200), (145, 215), (135, 235), (120, 245)
    ]
    draw.line(t3_points, fill=(240, 245, 235), width=4)

    # Armyworm chewed notches at leaf perimeter
    draw.ellipse([115, 170, 138, 195], fill=(240, 245, 240))  # Chewed notch 1
    draw.ellipse([250, 230, 275, 255], fill=(240, 245, 240))  # Chewed notch 2

    leaf.save(output_path, "JPEG", quality=92)


def create_spider_mites_sample(output_path: str):
    """Generates leaf with dense micro-stippling speckles, bronzed lamina, and fine webbing."""
    leaf = generate_base_leaf()
    arr = np.asarray(leaf, dtype=np.float32)

    # 1. Apply overall bronzing to leaf tissue (mites turn leaves dull golden/bronze)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    is_leaf = ~((r > 200) & (g > 200) & (b > 200))
    arr[is_leaf, 0] = np.clip(arr[is_leaf, 0] * 1.35 + 20, 0, 220)  # Boost red/bronze
    arr[is_leaf, 1] = np.clip(arr[is_leaf, 1] * 0.92, 0, 210)       # Drop pure green
    arr[is_leaf, 2] = np.clip(arr[is_leaf, 2] * 0.45, 0, 80)

    bronzed_leaf = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(bronzed_leaf)

    np.random.seed(303)
    # 2. Dense micro-stippling (700+ chlorotic pinhole speckles)
    for _ in range(750):
        px = int(np.random.normal(190, 52))
        py = int(np.random.normal(190, 80))
        if 110 < px < 270 and 80 < py < 320:
            c = (235, 230, 150)
            draw.point((px, py), fill=c)
            if np.random.rand() > 0.4:
                draw.point((px + 1, py), fill=c)
                draw.point((px, py + 1), fill=c)

    # 3. Fine silken mite webbing lines
    for _ in range(8):
        wx1 = np.random.randint(140, 220)
        wy1 = np.random.randint(150, 260)
        wx2 = wx1 + np.random.randint(-30, 30)
        wy2 = wy1 + np.random.randint(-25, 25)
        draw.line([(wx1, wy1), (wx2, wy2)], fill=(230, 235, 230), width=1)

    bronzed_leaf.save(output_path, "JPEG", quality=92)


def generate_all_samples():
    out_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(out_dir, exist_ok=True)

    samples = {
        "sample_healthy.jpg": create_healthy_sample,
        "sample_early_blight.jpg": create_early_blight_sample,
        "sample_common_rust.jpg": create_common_rust_sample,
        "sample_late_blight.jpg": create_late_blight_sample,
        "sample_chlorosis_stress.jpg": create_chlorosis_stress_sample,
        "sample_aphids.jpg": create_aphids_sample,
        "sample_leaf_miner.jpg": create_leaf_miner_sample,
        "sample_spider_mites.jpg": create_spider_mites_sample
    }

    for filename, fn in samples.items():
        filepath = os.path.join(out_dir, filename)
        fn(filepath)
        print(f"Generated sample: {filename}")


if __name__ == "__main__":
    generate_all_samples()
