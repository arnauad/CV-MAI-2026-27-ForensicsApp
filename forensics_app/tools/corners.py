from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Lab 5"
    description = "Mark FAST or Harris corners with an adjustable peak threshold."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        method = simpledialog.askstring("Corner detection", "Method (fast or harris):", initialvalue="fast", parent=parent)
        if method is None:
            return None
        method = method.strip().lower()
        if method not in ("fast", "harris"):
            return ToolResult(message="Unknown method. Use 'fast' or 'harris'.")
        threshold = simpledialog.askfloat("Corner detection", "Peak threshold (relative, 0 to 1):", initialvalue=0.1, minvalue=0, maxvalue=1, parent=parent)
        if threshold is None:
            return None
        pixels = np.asarray(document.current.convert("L")) / 255.0
        response = feature.corner_fast(pixels) if method == "fast" else feature.corner_harris(pixels)
        corner_points = feature.corner_peaks(response, threshold_rel=threshold)
        output = document.current.convert("RGB")
        draw = ImageDraw.Draw(output)
        for row, col in corner_points:
            draw.ellipse((col - 4, row - 4, col + 4, row + 4), outline=(255, 0, 0))
        return ToolResult(
            image=output,
            message=f"Marked {len(corner_points)} {method} corners in red.",
            details={"Operation": "Corner detection", "Method": method, "Threshold": threshold, "Corners": len(corner_points)},
        )
