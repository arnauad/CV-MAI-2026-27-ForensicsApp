"""Set 6.1: detect faces and five facial landmarks with the pretrained YuNet model."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
import urllib.request

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "Set6_materials" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
LANDMARK_NAMES = ["Right eye", "Left eye", "Nose", "Right mouth corner", "Left mouth corner"]


def load_model() -> Path:
    """Download the pretrained YuNet model if it is not available locally."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 100_000:
        raise RuntimeError("Model download failed or the file is incomplete.")
    return MODEL_PATH


class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    title = "Face Detection (YuNet)"
    category = "Lab 6"
    description = "Detect faces and facial landmarks with the pretrained YuNet CNN."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        score_threshold = simpledialog.askfloat(
            "Face Detection", "Confidence threshold (0 to 1):",
            initialvalue=0.6, minvalue=0, maxvalue=1, parent=parent,
        )
        if score_threshold is None:
            return None
        nms_threshold = simpledialog.askfloat(
            "Face Detection", "Non-maximum suppression threshold (0 to 1):",
            initialvalue=0.3, minvalue=0, maxvalue=1, parent=parent,
        )
        if nms_threshold is None:
            return None

        # OpenCV works on BGR arrays
        image = cv2.cvtColor(np.asarray(document.current.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            str(load_model()),
            "",
            (width, height),
            score_threshold,
            nms_threshold,
            5000,  # Maximum number of candidates
        )
        _, faces = detector.detect(image)
        if faces is None:
            faces = []
        result = image.copy()
        details: dict[str, object] = {
            "Operation": "Face Detection (YuNet)",
            "Image size": f"{width} x {height}",
            "Faces detected": len(faces),
        }

        for i, face in enumerate(faces):
            # Face bounding box
            x, y, w, h = face[:4].astype(int)
            confidence = float(face[14])
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                result, f"Face: {confidence:.2f}", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            details[f"Face {i + 1}"] = f"box ({x}, {y}, {w}, {h}), confidence {confidence:.2f}"

            # Five facial landmarks
            landmarks = face[4:14].reshape(5, 2).astype(int)
            for (px, py), name in zip(landmarks, LANDMARK_NAMES):
                cv2.circle(result, (px, py), 4, (255, 0, 0), -1)
                details[f"Face {i + 1} - {name}"] = f"({px}, {py})"

        message = f"Detected {len(faces)} face(s)." if len(faces) else "No faces detected."
        return ToolResult(
            image=Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)),
            message=message,
            details=details,
        )
