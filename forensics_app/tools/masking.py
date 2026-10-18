"""Keep the bright pixels of the image and fill the rest from another image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

THRESHOLD = 135
OPEN_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Lab 2"
    description = f"Keep pixels brighter than {THRESHOLD} and fill the rest from another image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        source = document.current.convert("RGB")

        filename = filedialog.askopenfilename(
            title="Choose the image to show through the mask", filetypes=OPEN_TYPES, parent=parent
        )
        if not filename:
            return None
        with Image.open(filename) as file:
            background = file.convert("RGB")
        if background.size != source.size:
            raise ValueError("The other image must have the same dimensions as the current image.")

        mask = np.asarray(source.convert("L")) > THRESHOLD  # shape (height, width)
        masked = np.where(mask[:, :, None], np.asarray(source), np.asarray(background))

        return ToolResult(
            image=Image.fromarray(masked.astype(np.uint8)),
            message=f"Kept pixels brighter than {THRESHOLD}; filled the rest from {Path(filename).name}.",
            details={"Operation": "Masking", "Threshold": THRESHOLD, "Kept pixels": f"{mask.mean():.1%}"},
        )
