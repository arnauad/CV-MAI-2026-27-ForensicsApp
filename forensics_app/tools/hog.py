"""Show the original image next to its HOG visualization for three orientations."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import exposure, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG extraction"
    category = "Lab 4"
    description = "Show the HOG visualization for three different orientation counts."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        orientations = []
        for i in (1, 2, 3):
            value = simpledialog.askinteger(
                "HOG extraction", f"Orientation {i}:", initialvalue=9, minvalue=1, parent=parent
            )
            if value is None:
                return None
            orientations.append(value)
        pixels_per_cell = simpledialog.askinteger(
            "HOG extraction", "Pixels per cell:", initialvalue=16, minvalue=1, parent=parent
        )
        if pixels_per_cell is None:
            return None
        cells_per_block = simpledialog.askinteger(
            "HOG extraction", "Cells per block:", initialvalue=2, minvalue=1, parent=parent
        )
        if cells_per_block is None:
            return None

        pixels = np.asarray(document.current.convert("RGB"))
        panels = [pixels]
        for orientation in orientations:
            _, hog_image = feature.hog(
                pixels,
                orientations=orientation,
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                visualize=True,
                channel_axis=2,
            )
            scaled = exposure.rescale_intensity(hog_image, out_range=(0, 255)).astype(np.uint8)
            panels.append(np.stack([scaled] * 3, axis=-1))

        return ToolResult(
            image=Image.fromarray(np.hstack(panels)),
            message=f"Showed the original image and its HOG for orientations {orientations}.",
            details={
                "Operation": "HOG extraction",
                "Orientations": orientations,
                "Pixels per cell": pixels_per_cell,
                "Cells per block": cells_per_block,
            },
        )
