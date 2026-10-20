"""Locate a reference person by sliding a window and comparing HOG descriptors."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person Detection"
    category = "Lab 4"
    description = "Find the window whose HOG descriptor best matches a reference person."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        filename = filedialog.askopenfilename(title="Open reference person image", parent=parent)
        if not filename:
            return None
        step_size = simpledialog.askinteger(
            "Person Detection", "Step size (pixels):", initialvalue=8, minvalue=1, parent=parent
        )
        if step_size is None:
            return None
        pixels_per_cell = simpledialog.askinteger(
            "Person Detection", "Pixels per cell:", initialvalue=8, minvalue=1, parent=parent
        )
        if pixels_per_cell is None:
            return None

        img = np.asarray(document.current.convert("L"))
        with Image.open(filename) as template_image:
            tmpl = np.asarray(template_image.convert("L"))
        size_person_x, size_person_y = img.shape
        size_window_x, size_window_y = tmpl.shape

        tmpl_fd = feature.hog(
            tmpl,
            orientations=9,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(2, 2),
        )

        coord, distances, products = [], [], []
        for i in range(0, size_person_x - size_window_x + 1, step_size):
            for j in range(0, size_person_y - size_window_y + 1, step_size):
                person_slice = img[i:i + size_window_x, j:j + size_window_y]
                img_fd = feature.hog(
                    person_slice,
                    orientations=9,
                    pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                    cells_per_block=(2, 2),
                )
                coord.append((i, j))
                distances.append(np.linalg.norm(tmpl_fd - img_fd))
                products.append(np.dot(img_fd, tmpl_fd))

        k_euclidean = np.argmin(distances)
        k_dot = np.argmax(products)

        output = document.current.convert("RGB")
        draw = ImageDraw.Draw(output)
        for k, color in ((k_euclidean, "red"), (k_dot, "lime")):
            # Same geometry as Rectangle((coord[k][1], coord[k][0]), size_window_y, size_window_x)
            x0, y0 = coord[k][1], coord[k][0]
            draw.rectangle((x0, y0, x0 + size_window_y - 1, y0 + size_window_x - 1), outline=color, width=3)

        return ToolResult(
            image=output,
            message="Red box: min Euclidean distance. Green box: max dot product.",
            details={
                "Operation": "Person Detection",
                "Windows evaluated": len(coord),
                "Window size (rows x cols)": f"{size_window_x} x {size_window_y}",
                "Step size": step_size,
                "Pixels per cell": pixels_per_cell,
                "Euclidean best (row, col)": coord[k_euclidean],
                "Euclidean distance": f"{distances[k_euclidean]:.3f}",
                "Dot product best (row, col)": coord[k_dot],
                "Dot product": f"{products[k_dot]:.3f}",
            },
        )
