"""Set 10: zero-shot classification with CLIP, on the open image or live from the webcam.

The "classifier" is just a list of sentences: each class text is put into a template
(e.g. "a photo of {}.") and CLIP picks the sentence closest to the image.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


# The model must be downloaded beforehand (~600 MB) from https://huggingface.co/openai/clip-vit-base-patch32
# into data/Set10_materials/clip-vit-base-patch32/. Only these files are needed (skip the TF/Flax copies):
#   config.json, preprocessor_config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
#   vocab.json, merges.txt, pytorch_model.bin
# For example:
#   hf download openai/clip-vit-base-patch32 --include "*.json" --include "*.txt" --include pytorch_model.bin \
#       --local-dir data/Set10_materials/clip-vit-base-patch32
MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "Set10_materials" / "clip-vit-base-patch32"

IMAGE_PRESETS: dict[str, tuple[str, list[str]]] = {
    "Animals": ("a photo of a {}.", ["cat", "dog", "tiger", "rabbit", "fox", "owl"]),
    "Counting cats": ("a photo of {}.", ["one cat", "two cats", "three cats", "no cats"]),
    "Image style": ("a {}.", ["photo", "painting", "cartoon", "pencil sketch"]),
    "Image origin": (
        "{}",
        ["a real photograph", "an AI-generated image", "a screenshot", "a scanned document", "a digital painting"],
    ),
    "Drinks": ("a photo of a cup of {}.", ["coffee", "tea", "soup", "hot chocolate"]),
}
WEBCAM_PRESETS: dict[str, tuple[str, list[str]]] = {
    "Classroom: people": (
        "a photo of {}.",
        ["one person", "two people", "three people", "a person holding a pen"],
    ),
    "Held objects": (
        "a photo of a person holding {}.",
        ["a pen", "a phone", "a cup", "a book", "nothing"],
    ),
    "Gestures": (
        "a photo of {}.",
        ["a person waving", "a person giving a thumbs up", "a person covering their face", "an empty room"],
    ),
    "Accessories": (
        "a photo of a person {}.",
        ["wearing glasses", "wearing a hat", "wearing headphones", "with nothing on their head"],
    ),
}


@lru_cache(maxsize=1)
def load_clip():
    """Load CLIP from disk once per session; return ``(model, processor)``."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if not (MODEL_DIR / "pytorch_model.bin").exists():
        raise RuntimeError(f"CLIP not found in {MODEL_DIR}. See the comment at the top of clip_classification.py.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))  # Keep the laptop responsive during live video
    model = CLIPModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    return model, processor


def build_prompts(template: str, labels: list[str]) -> list[str]:
    """Put each class text where ``{}`` is in the template."""
    return [template.replace("{}", label) for label in labels]


def _as_tensor(output):
    # Newer transformers versions may return an output object instead of a tensor
    return output if hasattr(output, "shape") else output.pooler_output


class ClipClassifier:
    """Zero-shot classifier: the text embeddings are computed once, then reused for every image."""

    def __init__(self, template: str, labels: list[str]) -> None:
        import torch

        self.torch = torch
        self.model, self.processor = load_clip()
        inputs = self.processor(text=build_prompts(template, labels), return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = _as_tensor(self.model.get_text_features(**inputs))
        self.text_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return one probability per class (softmax over scaled cosine similarities)."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            embedding = _as_tensor(self.model.get_image_features(**inputs))
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * embedding @ self.text_embeddings.T
        return logits.softmax(dim=-1)[0].numpy()


class ClassEditor(tk.Toplevel):
    """Dialog to choose a prompt template and any number of class texts, starting from presets."""

    def __init__(self, parent: tk.Misc, title: str, presets: dict[str, tuple[str, list[str]]]) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.presets = presets
        self.result: tuple[str, list[str]] | None = None
        self.entries: list[tuple[ttk.Frame, tk.StringVar]] = []

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Preset:").grid(row=0, column=0, sticky="w")
        self.preset = ttk.Combobox(body, values=list(presets), state="readonly", width=32)
        self.preset.grid(row=0, column=1, sticky="ew", pady=2)
        self.preset.bind("<<ComboboxSelected>>", lambda _event: self._load_preset(self.preset.get()))

        ttk.Label(body, text="Template:").grid(row=1, column=0, sticky="w")
        self.template = tk.StringVar()
        ttk.Entry(body, textvariable=self.template, width=40).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(body, text="{} is replaced by each class text; use just {} to send the text as is.",
                  foreground="#555").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(body, text="Classes:").grid(row=3, column=0, sticky="nw")
        self.class_frame = ttk.Frame(body)
        self.class_frame.grid(row=3, column=1, sticky="ew")
        ttk.Button(body, text="+ Add class", command=self._add_class).grid(row=4, column=1, sticky="w", pady=4)

        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Run", command=self._accept).pack(side="right", padx=(0, 6))
        body.columnconfigure(1, weight=1)

        first = next(iter(presets))
        self.preset.set(first)
        self._load_preset(first)
        self.bind("<Return>", lambda _event: self._accept())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.grab_set()

    def _load_preset(self, name: str) -> None:
        template, labels = self.presets[name]
        self.template.set(template)
        for row, _ in self.entries:
            row.destroy()
        self.entries.clear()
        for label in labels:
            self._add_class(label)

    def _add_class(self, text: str = "") -> None:
        row = ttk.Frame(self.class_frame)
        row.pack(fill="x", pady=1)
        value = tk.StringVar(value=text)
        entry = ttk.Entry(row, textvariable=value, width=36)
        entry.pack(side="left", fill="x", expand=True)
        item = (row, value)
        ttk.Button(row, text="✕", width=3, command=lambda: self._remove_class(item)).pack(side="left", padx=(4, 0))
        self.entries.append(item)
        if not text:
            entry.focus_set()

    def _remove_class(self, item: tuple[ttk.Frame, tk.StringVar]) -> None:
        item[0].destroy()
        self.entries.remove(item)

    def _accept(self) -> None:
        template = self.template.get().strip()
        labels = [value.get().strip() for _, value in self.entries if value.get().strip()]
        if "{}" not in template:
            messagebox.showerror("CLIP", "The template must contain {}.", parent=self)
            return
        if len(labels) < 2:
            messagebox.showerror("CLIP", "Write at least two classes.", parent=self)
            return
        self.result = (template, labels)
        self.destroy()


def ask_classes(parent: tk.Misc, title: str, presets: dict[str, tuple[str, list[str]]]):
    """Return ``(template, labels)`` or ``None`` if cancelled."""
    dialog = ClassEditor(parent, title, presets)
    parent.wait_window(dialog)
    return dialog.result


def probability_details(labels: list[str], probabilities: np.ndarray) -> dict[str, str]:
    order = np.argsort(-probabilities)
    return {f"  {labels[i]}": f"{probabilities[i]:.1%}" for i in order}


def draw_bar_chart(image: Image.Image, labels: list[str], probabilities: np.ndarray, template: str) -> Image.Image:
    """Image next to a horizontal bar chart of class probabilities (as in the Set 10 notebook)."""
    from matplotlib.figure import Figure

    order = np.argsort(probabilities)
    figure = Figure(figsize=(10, max(3.2, 0.45 * len(labels))), layout="constrained")
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    left.imshow(image.convert("RGB"))
    left.axis("off")
    right.barh(np.array(labels)[order], probabilities[order], color="#3b6ea8")
    right.set_xlim(0, 1)
    right.set_xlabel("probability")
    right.set_title(f'template: "{template}"')
    for i, value in enumerate(probabilities[order]):
        right.text(min(value + 0.02, 0.9), i, f"{value:.2f}", va="center")
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        with Image.open(buffer) as chart:
            return chart.convert("RGB")


def draw_overlay(frame: np.ndarray, labels: list[str], probabilities: np.ndarray) -> np.ndarray:
    """Draw one probability bar per class on a copy of a BGR frame; the best class is green."""
    result = frame.copy()
    best = int(np.argmax(probabilities))
    bar_width, row_height = 180, 26
    text_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for label in labels)
    panel = result[: 10 + row_height * len(labels), : 30 + text_width + bar_width]
    panel[:] = (panel * 0.45).astype(np.uint8)  # Darken the background so the text is readable
    for i, (label, probability) in enumerate(zip(labels, probabilities)):
        y = 10 + i * row_height
        color = (0, 220, 0) if i == best else (230, 180, 60)
        x = 20 + text_width
        cv2.putText(result, label, (10, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(result, (x, y + 4), (x + bar_width, y + 20), (90, 90, 90), 1)
        cv2.rectangle(result, (x, y + 4), (x + int(bar_width * probability), y + 20), color, -1)
        cv2.putText(result, f"{probability:.0%}", (x + bar_width - 42, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


class ClipClassificationTool(ForensicsTool):
    tool_id = "clip_classification"
    title = "CLIP zero-shot"
    category = "Lab 10"
    description = "Classify the open image with CLIP using any number of classes written as text."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = ask_classes(parent, "CLIP zero-shot classification", IMAGE_PRESETS)
        if choice is None:
            return None
        template, labels = choice
        classifier = ClipClassifier(template, labels)
        start = time.perf_counter()
        probabilities = classifier.predict(document.current)
        seconds = time.perf_counter() - start
        best = int(probabilities.argmax())

        # Keep the open image unchanged so it can be classified again with other texts
        chart = ImageTk.PhotoImage(draw_bar_chart(document.current, labels, probabilities, template))
        window = tk.Toplevel(parent)
        window.title(f"CLIP · {labels[best]}")
        label = ttk.Label(window, image=chart)
        label.image = chart  # Keep a reference so it is not garbage-collected
        label.pack(padx=8, pady=8)

        return ToolResult(
            message=f"CLIP: {labels[best]} ({probabilities[best]:.1%})",
            details={
                "Model": MODEL_NAME,
                "Template": template,
                "Prediction": labels[best],
                "Probability": f"{probabilities[best]:.1%}",
                "Classes": len(labels),
                **probability_details(labels, probabilities),
                "Inference time": f"{seconds * 1000:.0f} ms",
            },
        )


class WebcamClipClassificationTool(ForensicsTool):
    tool_id = "webcam_clip_classification"
    title = "CLIP zero-shot (Webcam)"
    category = "Lab 10"
    description = "Classify the laptop webcam in real time with CLIP and your own class texts."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        choice = ask_classes(parent, "CLIP zero-shot classification (Webcam)", WEBCAM_PRESETS)
        if choice is None:
            return None
        template, labels = choice
        classifier = ClipClassifier(template, labels)

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("Could not open the webcam.")

        window = tk.Toplevel(parent)
        window.title("CLIP zero-shot (Webcam)")
        video = ttk.Label(window)
        video.pack(padx=8, pady=8)
        info = tk.StringVar(value="Starting webcam…")
        ttk.Label(window, textvariable=info).pack()
        buttons = ttk.Frame(window, padding=8)
        buttons.pack()

        state: dict[str, object] = {"frame": None, "probabilities": None, "captured": None, "seconds": 0.0}

        def show_frame() -> None:
            if not window.winfo_exists():
                return
            ok, frame = capture.read()
            if ok:
                frame = cv2.flip(frame, 1)  # Mirror the view
                start = time.perf_counter()
                probabilities = classifier.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                seconds = time.perf_counter() - start
                previous = state["probabilities"]
                if previous is not None:  # Smooth over frames so the bars do not flicker
                    probabilities = 0.6 * previous + 0.4 * probabilities
                result = draw_overlay(frame, labels, probabilities)
                state.update(frame=result, probabilities=probabilities, seconds=seconds)
                best = int(np.argmax(probabilities))
                info.set(f"{labels[best]} ({probabilities[best]:.0%}) · {seconds * 1000:.0f} ms per frame")
                photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
                video.photo = photo  # Keep a reference so it is not garbage-collected
                video.configure(image=photo)
            window.after(10, show_frame)

        def capture_frame() -> None:
            if state["frame"] is not None:
                state["captured"] = (state["frame"], state["probabilities"], state["seconds"])
                window.destroy()

        ttk.Button(buttons, text="Capture frame", command=capture_frame).pack(side="left")
        ttk.Button(buttons, text="Close", command=window.destroy).pack(side="left", padx=(6, 0))
        window.bind("<space>", lambda _event: capture_frame())

        show_frame()
        parent.wait_window(window)
        capture.release()

        if state["captured"] is None:
            return None
        frame, probabilities, seconds = state["captured"]
        best = int(np.argmax(probabilities))
        height, width = frame.shape[:2]
        return ToolResult(
            image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            message=f"Captured webcam frame · CLIP: {labels[best]} ({probabilities[best]:.1%})",
            details={
                "Operation": "CLIP zero-shot (Webcam)",
                "Model": MODEL_NAME,
                "Image size": f"{width} x {height}",
                "Template": template,
                "Prediction": labels[best],
                "Probability": f"{probabilities[best]:.1%}",
                **probability_details(labels, probabilities),
                "Inference time": f"{seconds * 1000:.0f} ms",
            },
        )
