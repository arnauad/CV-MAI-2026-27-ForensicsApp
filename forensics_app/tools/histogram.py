"""Plot how often each intensity (0-255) appears in the R, G and B channels."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

HEIGHT = 300


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Starter tools"
    description = "Show the frequency of each pixel intensity for the R, G and B channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))  # shape (height, width, 3)
        counts = [np.histogram(pixels[:, :, c], bins=256, range=(0, 256))[0] for c in range(3)]
        peak = max(c.max() for c in counts)

        canvas = tk.Canvas(tk.Toplevel(parent), width=512, height=HEIGHT, bg="white") # tk.Toplevel creates a new separate window
        canvas.pack()
        for count, color in zip(counts, ["red", "green", "blue"]):
            points = [(2 * x, HEIGHT - y * HEIGHT / peak) for x, y in enumerate(count)]
            canvas.create_line(points, fill=color)

        return ToolResult(
            message="Opened the R, G and B histograms in a new window.",
            details={"Operation": "Histogram", **{f"Mean {n}": f"{pixels[:, :, i].mean():.1f}" for i, n in enumerate("RGB")}},
        )
