from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image
from skimage import feature, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CrossCorrelationTool(ForensicsTool):
    tool_id = "cross_correlation"
    title = "Cross-correlation"
    category = "Starter tools"
    description = "Show normalized cross-correlation with a template image."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(title="Open template image", parent=parent)
        if not filename:
            return None
        image = np.asarray(document.current.convert("L"))
        with Image.open(filename) as template_image:
            template = np.asarray(template_image.convert("L"))
        correlation = feature.match_template(image, template, pad_input=True)
        output = ((np.clip(correlation, -1, 1) + 1) * 127.5).round().astype(np.uint8)
        return ToolResult(
            image=Image.fromarray(output),
            message="Showed the normalized cross-correlation map.",
            details={"Operation": "Cross-correlation", "Best score": f"{correlation.max():.3f}"},
        )


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Template matching"
    category = "Starter tools"
    description = "Mark template matches with an adjustable angle and threshold."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(title="Open template image", parent=parent)
        if not filename:
            return None
        angle = simpledialog.askfloat("Template matching", "Angle (degrees):", initialvalue=0, parent=parent)
        if angle is None:
            return None
        threshold = simpledialog.askfloat("Template matching", "Threshold (-1 to 1):", initialvalue=0.8, minvalue=-1, maxvalue=1, parent=parent)
        if threshold is None:
            return None
        image = np.asarray(document.current.convert("L")) / 255.0
        with Image.open(filename) as template_image:
            template = np.asarray(template_image.convert("L")) / 255.0
        template = transform.rotate(template, angle)
        correlation = feature.match_template(image, template, pad_input=True)
        matches = correlation >= threshold
        output = np.array(document.current.convert("RGB"))
        output[matches] = [255, 0, 0]
        return ToolResult(
            image=Image.fromarray(output),
            message="Marked template match centers in red.",
            details={"Operation": "Template matching", "Angle": angle, "Threshold": threshold, "Marked pixels": int(matches.sum())},
        )
