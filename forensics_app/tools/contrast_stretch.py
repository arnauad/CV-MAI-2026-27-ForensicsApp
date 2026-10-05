"""Stretch the 2nd-98th percentile intensity range to the full 0-255 range."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import exposure

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Starter tools"
    description = "Stretch the 2nd-98th percentile intensity range to the full 0-255 range."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))
        p2, p98 = np.percentile(pixels, (2, 98))
        stretched = exposure.rescale_intensity(pixels, in_range=(p2, p98), out_range=(0, 255))
        return ToolResult(
            image=Image.fromarray(stretched.astype(np.uint8)),
            message="Stretched the contrast.",
            details={"Operation": "Contrast stretching", "Input range (p2-p98)": f"{p2:.0f}-{p98:.0f}"},
        )
