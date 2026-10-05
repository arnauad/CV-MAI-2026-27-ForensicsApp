"""Keep only the bright pixels of the image and black out the rest."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

THRESHOLD = 135


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Starter tools"
    description = f"Keep pixels brighter than {THRESHOLD} and set the rest to black."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))  # shape (height, width, 3)
        brightness = np.asarray(document.current.convert("L"))  # shape (height, width)
        mask = brightness > THRESHOLD
        masked = pixels * mask[:, :, None]  # broadcast the mask over the 3 channels
        return ToolResult(
            image=Image.fromarray(masked.astype(np.uint8)),
            message=f"Kept pixels brighter than {THRESHOLD}.",
            details={"Operation": "Masking", "Threshold": THRESHOLD, "Kept pixels": f"{mask.mean():.1%}"},
        )
