"""Set 5.3: match ORB or SIFT descriptors between two images."""

from pathlib import Path
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .keypoints import compute_keypoints_and_descriptors


IMAGE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]


def ask_matching_settings(parent, title):
    """Return shared parameters, or None when a dialog is cancelled."""
    method = simpledialog.askstring(title, "Method (orb or sift):", initialvalue="orb", parent=parent)
    if method is None:
        return None
    method = method.strip().lower()
    if method not in ("orb", "sift"):
        raise ValueError("Unknown method. Use 'orb' or 'sift'.")
    count = simpledialog.askinteger(title, "Maximum keypoints:", initialvalue=500, minvalue=1, parent=parent)
    if count is None:
        return None
    ratio = simpledialog.askfloat(
        title, "Match ratio (0 to 1; lower is stricter):",
        initialvalue=0.8, minvalue=0, maxvalue=1, parent=parent,
    )
    if ratio is None:
        return None
    return method, count, ratio


def extract_features(image: Image.Image, method: str, count: int):
    """Reuse Set 5.2; images without usable features return None."""
    if method not in ("orb", "sift"):
        raise ValueError("Unknown method. Use 'orb' or 'sift'.")
    pixels = np.asarray(image.convert("L")) / 255.0
    if min(pixels.shape) < 16 or np.ptp(pixels) == 0:
        return None
    try:
        keypoints, descriptors = compute_keypoints_and_descriptors(pixels, method, count)
    except RuntimeError:
        # ORB and SIFT raise RuntimeError when no features are found.
        return None
    if len(keypoints) == 0:
        return None
    return keypoints, descriptors


def match_features(first, second, method: str, ratio: float) -> np.ndarray:
    """Keep mutual nearest neighbours that pass the descriptor ratio test."""
    # The ratio test needs two candidate descriptors in the second image.
    if first is None or second is None or len(second[1]) < 2:
        return np.empty((0, 2), dtype=int)
    return feature.match_descriptors(
        first[1], second[1],
        metric="hamming" if method == "orb" else "euclidean",
        cross_check=True, max_ratio=ratio,
    )


def draw_matches(first_image, second_image, first_points, second_points, matches):
    """Place images side by side and join matching (row, column) locations."""
    output = Image.new("RGB", (first_image.width + second_image.width,
                               max(first_image.height, second_image.height)))
    output.paste(first_image.convert("RGB"), (0, 0))
    output.paste(second_image.convert("RGB"), (first_image.width, 0))
    draw = ImageDraw.Draw(output)
    colors = ("red", "lime", "cyan", "yellow", "magenta")
    for index, (i, j) in enumerate(matches):
        row1, col1 = first_points[i]
        row2, col2 = second_points[j]
        x1, y1 = float(col1), float(row1)
        x2, y2 = float(col2) + first_image.width, float(row2)
        color = colors[index % len(colors)]
        draw.line((x1, y1, x2, y2), fill=color)
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color)
    return output


class ImageMatchingTool(ForensicsTool):
    tool_id = "image_matching"
    title = "Match second image"
    category = "Lab 5"
    description = "Match ORB or SIFT descriptors with a second image."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(title="Open second image", filetypes=IMAGE_TYPES, parent=parent)
        if not filename:
            return None
        settings = ask_matching_settings(parent, self.title)
        if settings is None:
            return None
        method, count, ratio = settings
        with Image.open(filename) as image:
            second_image = image.convert("RGB")
        first = extract_features(document.current, method, count)
        second = extract_features(second_image, method, count)
        matches = match_features(first, second, method, ratio)
        details = {
            "Operation": "Image matching", "Second image": Path(filename).name,
            "Method": method.upper(), "Maximum keypoints": count, "Match ratio": ratio,
            "First keypoints": len(first[0]) if first is not None else 0,
            "Second keypoints": len(second[0]) if second is not None else 0,
            "Matches": len(matches),
        }
        if len(matches) == 0:
            return ToolResult(message="No descriptor matches found. Try another image or match ratio.", details=details)
        return ToolResult(
            image=draw_matches(document.current, second_image, first[0], second[0], matches),
            message=f"Drew {len(matches)} {method.upper()} descriptor matches.", details=details,
        )
