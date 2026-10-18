from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Lab 3"
    description = "Detect edges with an adjustable Canny sigma."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        sigma = simpledialog.askfloat("Canny edge detection", "Sigma:", initialvalue=2, minvalue=0, parent=parent)
        if sigma is None:
            return None
        pixels = np.asarray(document.current.convert("L"))
        edges = feature.canny(pixels, sigma=sigma, low_threshold=0.1, high_threshold=0.2)
        return ToolResult(
            image=Image.fromarray(edges.astype(np.uint8) * 255),
            message="Applied Canny edge detection.",
            details={"Operation": "Canny edge detection", "Sigma": sigma},
        )


class EdgeVisualizationTool(ForensicsTool):
    tool_id = "edge_visualization"
    title = "Edge visualization"
    category = "Lab 3"
    description = "Draw detected edges in red over the image."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        sigma = simpledialog.askfloat("Edge visualization", "Sigma:", initialvalue=2, minvalue=0, parent=parent)
        if sigma is None:
            return None
        pixels = np.asarray(document.current.convert("L")) / 255.0
        edges = feature.canny(pixels, sigma=sigma, low_threshold=0.1, high_threshold=0.2)
        output = np.array(document.current.convert("RGB"))
        output[edges] = [255, 0, 0]
        return ToolResult(
            image=Image.fromarray(output),
            message="Drew detected edges in red over the image.",
            details={"Operation": "Edge visualization", "Sigma": sigma},
        )
