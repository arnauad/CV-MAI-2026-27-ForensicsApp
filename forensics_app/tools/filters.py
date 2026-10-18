from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class GaussianTool(ForensicsTool):
    tool_id = "gaussian"
    title = "Gaussian filter"
    category = "Lab 3"
    description = "Blur the image with an adjustable Gaussian sigma."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        sigma = simpledialog.askfloat("Gaussian filter", "Sigma:", initialvalue=2, minvalue=0, parent=parent)
        if sigma is None:
            return None
        pixels = np.asarray(document.current.convert("L"))
        output = filters.gaussian(pixels, sigma=sigma)
        return ToolResult(
            image=Image.fromarray((output * 255).round().astype(np.uint8)),
            message="Applied Gaussian filtering.",
            details={"Operation": "Gaussian filter", "Sigma": sigma},
        )


class MedianTool(ForensicsTool):
    tool_id = "median"
    title = "Median filter"
    category = "Lab 3"
    description = "Reduce isolated noise with a median filter."

    def run(self, parent, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        pixels = np.asarray(document.current.convert("L"))
        output = filters.median(pixels)
        return ToolResult(
            image=Image.fromarray(output),
            message="Applied median filtering.",
            details={"Operation": "Median filter"},
        )


class SobelTool(ForensicsTool):
    tool_id = "sobel"
    title = "Sobel filter"
    category = "Lab 3"
    description = "Highlight edges with a Sobel filter."

    def run(self, parent, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        pixels = np.asarray(document.current.convert("L"))
        output = filters.sobel(pixels)
        return ToolResult(
            image=Image.fromarray((np.clip(output, 0, 1) * 255).round().astype(np.uint8)),
            message="Applied Sobel edge filtering.",
            details={"Operation": "Sobel filter"},
        )


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Sharpen"
    category = "Lab 3"
    description = "Sharpen the image via unsharp masking."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        sigma = simpledialog.askfloat("Sharpen", "Sigma:", initialvalue=2, minvalue=0, parent=parent)
        if sigma is None:
            return None
        amount = simpledialog.askfloat("Sharpen", "Amount:", initialvalue=1.5, minvalue=0, parent=parent)
        if amount is None:
            return None
        pixels = np.asarray(document.current.convert("L"))
        output = filters.unsharp_mask(pixels, radius=sigma, amount=amount)
        return ToolResult(
            image=Image.fromarray((output * 255).round().astype(np.uint8)),
            message="Applied unsharp masking.",
            details={"Operation": "Sharpen", "Sigma": sigma, "Amount": amount},
        )
