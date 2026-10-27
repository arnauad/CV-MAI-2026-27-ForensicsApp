"""Detect keypoints and extract their descriptors with ORB or SIFT."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature
from skimage.feature import SIFT

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_keypoints_and_descriptors(
    pixels: np.ndarray,
    method: str,
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "orb":
        extractor = feature.ORB(n_keypoints=n_keypoints)
        extractor.detect_and_extract(pixels)
        return extractor.keypoints, extractor.descriptors
    if method == "sift":
        extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min)
        extractor.detect_and_extract(pixels)
        return extractor.keypoints[:n_keypoints], extractor.descriptors[:n_keypoints]
    raise ValueError(f"Unknown method {method!r}; use 'orb' or 'sift'.")


class KeypointDescriptorTool(ForensicsTool):
    tool_id = "keypoint_descriptors"
    title = "Keypoints & descriptors"
    category = "Lab 5"
    description = "Detect ORB or SIFT keypoints and extract their descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        method = simpledialog.askstring(
            "Keypoints & descriptors", "Method (orb or sift):", initialvalue="orb", parent=parent
        )
        if method is None:
            return None
        method = method.strip().lower()
        if method not in ("orb", "sift"):
            return ToolResult(message="Unknown method. Use 'orb' or 'sift'.")
        n_keypoints = simpledialog.askinteger(
            "Keypoints & descriptors", "Number of keypoints:", initialvalue=500, minvalue=1, parent=parent
        )
        if n_keypoints is None:
            return None
        n_octaves, n_scales, sigma_min = 8, 3, 1.6
        if method == "sift":
            n_octaves = simpledialog.askinteger(
                "Keypoints & descriptors", "Number of octaves:", initialvalue=8, minvalue=1, parent=parent
            )
            if n_octaves is None:
                return None
        if method == "sift":
            n_scales = simpledialog.askinteger(
                "Keypoints & descriptors", "Number of scales:", initialvalue=3, minvalue=1, parent=parent,
            )
            if n_scales is None:
                return None
        if method == "sift":
            sigma_min = simpledialog.askfloat(
                "Keypoints & descriptors", "Sigma min:", initialvalue=1.6, minvalue=0.01, parent=parent
            )
            if sigma_min is None:
                return None

        pixels = np.asarray(document.current.convert("L")) / 255.0
        try:
            keypoints, descriptors = compute_keypoints_and_descriptors(
                pixels, method, n_keypoints, n_octaves, n_scales, sigma_min
            )
        except RuntimeError as error:
            return ToolResult(message=f"{method.upper()} failed: {error}")

        output = document.current.convert("RGB")
        draw = ImageDraw.Draw(output)
        for row, col in keypoints:
            draw.ellipse((col - 4, row - 4, col + 4, row + 4), outline=(255, 0, 0))
        details = {"Operation": "Keypoints & descriptors", "Method": method, "Number of keypoints": n_keypoints}
        if method == "sift":
            details["Number of octaves"] = n_octaves
            details["Number of scales"] = n_scales
            details["Sigma min"] = sigma_min
        details.update(
            {
                "Keypoints found": len(keypoints),
                "Descriptor shape": descriptors.shape,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )
        return ToolResult(
            image=output,
            message=f"Marked {len(keypoints)} {method.upper()} keypoints in red.",
            details=details,
        )
