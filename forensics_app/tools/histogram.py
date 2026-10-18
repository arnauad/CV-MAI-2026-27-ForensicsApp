"""Plot how often each intensity (0-255) appears in the R, G and B channels."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

WIDTH = 512
HEIGHT = 300
MARGIN = 30


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Lab 2"
    description = "Show the frequency of each pixel intensity for the R, G and B channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))  # shape (height, width, 3)
        counts = [np.histogram(pixels[:, :, c], bins=256, range=(0, 256))[0] for c in range(3)]
        peak = max(c.max() for c in counts)
        plot_w, plot_h = WIDTH - MARGIN, HEIGHT - MARGIN

        canvas = tk.Canvas(tk.Toplevel(parent), width=WIDTH, height=HEIGHT, bg="white") # tk.Toplevel creates a new separate window
        canvas.pack()
        canvas.create_line(MARGIN, 0, MARGIN, plot_h, MARGIN + plot_w, plot_h)  # y and x axes

        for frac in (0, 0.25, 0.5, 0.75, 1.0):  # x-axis ticks: intensity 0-255
            x = MARGIN + frac * plot_w
            canvas.create_text(x, plot_h + 10, text=str(round(frac * 255)), font=("", 8))
        for frac in (0, 0.5, 1.0):  # y-axis ticks: pixel count
            canvas.create_text(MARGIN - 18, plot_h - frac * plot_h, text=str(round(frac * peak)), font=("", 8))

        for count, color in zip(counts, ["red", "green", "blue"]):
            points = [(MARGIN + x * plot_w / 255, plot_h - y * plot_h / peak) for x, y in enumerate(count)]
            canvas.create_line(points, fill=color)

        return ToolResult(
            message="Opened the R, G and B histograms in a new window.",
            details={"Operation": "Histogram", **{f"Mean {n}": f"{pixels[:, :, i].mean():.1f}" for i, n in enumerate("RGB")}},
        )
