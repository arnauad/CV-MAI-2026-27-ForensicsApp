"""Show the red, green and blue channels of the image side by side."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Starter tools"
    description = "Show the red, green and blue channels of the image side by side."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))  # shape (height, width, 3)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        output = Image.fromarray(np.hstack([red, green, blue]))
        return ToolResult(
            image=output,
            message="Show the red, green and blue channels of the image side by side.",
            details={"Operation": "Channel split", "Order": "R | G | B"},
        )
