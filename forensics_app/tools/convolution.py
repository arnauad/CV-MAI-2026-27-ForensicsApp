import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Starter tools"
    description = "Apply horizontal, vertical or box convolution."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        window = tk.Toplevel(parent)
        window.title("Convolution")
        window.transient(parent)
        selected = tk.StringVar(master=window)

        def choose(direction):
            selected.set(direction)
            window.destroy()

        for direction in ("Horizontal", "Vertical", "Box"):
            ttk.Button(window, text=direction, command=lambda d=direction: choose(d)).pack(
                side="left", padx=10, pady=15
            )
        window.grab_set()
        window.wait_window()
        direction = selected.get()
        if not direction:
            return None

        kernel_horizontal = np.ones((1, 15), dtype=np.int32) / 15
        kernel_vertical = np.transpose(kernel_horizontal)
        kernel_box = np.ones((15, 15)) / (15 * 15)
        kernel = {"Horizontal": kernel_horizontal, "Vertical": kernel_vertical, "Box": kernel_box}[direction]
        pixels = np.asarray(document.current.convert("L"), dtype=np.float64)
        output = ndimage.convolve(pixels, kernel)

        return ToolResult(
            image=Image.fromarray(np.clip(output, 0, 255).round().astype(np.uint8)),
            message=f"Applied {direction.lower()} convolution.",
            details={"Operation": "Convolution", "Direction": direction, "Kernel": f"{kernel.shape[0]} x {kernel.shape[1]}"},
        )
