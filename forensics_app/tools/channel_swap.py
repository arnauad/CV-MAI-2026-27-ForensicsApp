"""Swap the red and blue channels of the image (RGB -> BGR)."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Starter tools"
    description = "Swap the red and blue channels (RGB -> BGR)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        pixels = np.asarray(document.current.convert("RGB"))  # shape (height, width, 3)
        swapped = pixels[:, :, ::-1]  # reverse the channel axis: RGB -> BGR
        return ToolResult(
            image=Image.fromarray(swapped),
            message="Swapped the channels: RGB -> BGR.",
            details={"Operation": "Channel swap", "Order": "BGR"},
        )
