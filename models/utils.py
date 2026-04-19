"""Shared drawing utilities for all model wrappers."""
from __future__ import annotations

from PIL import ImageDraw, ImageFont

# Distinct colors for up to 20 classes (RGB tuples)
CLASS_COLORS: list[tuple[int, int, int]] = [
    (255, 56,  56),   # red
    (255, 157,  51),  # orange
    (255, 230,  51),  # yellow
    (51,  255,  51),  # green
    (51,  255, 255),  # cyan
    (51,  153, 255),  # blue
    (178,  51, 255),  # purple
    (255,  51, 178),  # pink
    (255, 255, 255),  # white
    (102, 255, 102),  # light green
    (102, 178, 255),  # light blue
    (255, 102, 102),  # light red
    (255, 204, 102),  # peach
    (102, 255, 204),  # mint
    (204, 102, 255),  # lavender
    (255, 102, 204),  # rose
    (0,   204, 204),  # teal
    (204, 204,   0),  # olive
    (204,   0,   0),  # dark red
    (0,   102, 204),  # dark blue
]


def draw_label(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    conf: float,
    color: tuple[int, int, int],
    line_width: int = 2,
) -> None:
    """Draw a bounding box and label badge on a PIL ImageDraw canvas."""
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    text = f"{label} {conf:.0%}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 3

    ty = max(0, y1 - th - pad * 2)
    # Badge background
    draw.rectangle(
        [x1, ty, x1 + tw + pad * 2, ty + th + pad * 2],
        fill=color,
    )
    # Text — use black for bright colors, white otherwise
    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
    draw.text((x1 + pad, ty + pad), text, fill=text_color, font=font)