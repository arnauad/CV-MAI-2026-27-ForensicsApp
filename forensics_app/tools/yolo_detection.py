"""Lab 9 object detection with an Ultralytics YOLO model (pretrained or trained in Colab)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog

from PIL import ImageDraw, ImageFont

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

PREDICTION_COLORS = ["red", "blue", "orange", "magenta", "cyan", "yellow", "purple", "brown"]
GROUND_TRUTH_COLOR = "lime"


class YoloDetectionTool(ForensicsTool):
    tool_id = "yolo_detection"
    title = "YOLO object detection"
    category = "Lab 9"
    description = (
        "Detect objects with a YOLO .pt model (e.g. yolo11n.pt or the blood-cell best.pt). "
        "Ground-truth boxes are drawn in green when the image has a YOLO label file."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            parent=parent, title="Choose a YOLO model", filetypes=[("YOLO weights", "*.pt")]
        )
        if not filename:
            return None
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise RuntimeError("Install the dependencies from requirements.txt.") from error

        model = YOLO(filename)
        image = document.current.convert("RGB")
        start = time.perf_counter()
        prediction = model.predict(image, verbose=False)[0]
        seconds = time.perf_counter() - start

        output = image.copy()
        draw = ImageDraw.Draw(output)
        font = ImageFont.load_default(size=max(12, image.width // 60))
        ground_truth = read_ground_truth(document.path, image.width, image.height)
        for _, box in ground_truth:
            draw.rectangle(box, outline=GROUND_TRUTH_COLOR, width=2)

        counts: Counter[str] = Counter()
        boxes = prediction.boxes
        for box, confidence, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            name = prediction.names[int(cls)]
            color = PREDICTION_COLORS[int(cls) % len(PREDICTION_COLORS)]
            counts[name] += 1
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 3, box[1] + 2), f"{name} {confidence:.2f}", fill=color, font=font)

        details = {
            "Model": Path(filename).name,
            "Detections": len(boxes),
            **{f"  {name}": count for name, count in sorted(counts.items())},
            "Inference time": f"{seconds * 1000:.0f} ms",
        }
        if ground_truth:
            details["Ground-truth boxes (green)"] = len(ground_truth)
        return ToolResult(
            image=output,
            message=f"{len(boxes)} objects detected with {Path(filename).name}.",
            details=details,
        )


def read_ground_truth(image_path: Path | None, width: int, height: int) -> list[tuple[int, tuple[float, ...]]]:
    """Read pixel boxes from the YOLO label file of a dataset image (.../images/x.jpg -> .../labels/x.txt)."""
    if image_path is None:
        return []
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            cls, x_center, y_center, w, h = map(float, line.split())
            box = ((x_center - w / 2) * width, (y_center - h / 2) * height,
                   (x_center + w / 2) * width, (y_center + h / 2) * height)
            boxes.append((int(cls), box))
    return boxes
