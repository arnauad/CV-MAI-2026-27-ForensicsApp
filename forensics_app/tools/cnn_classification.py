"""Lab 8 inference and feature projections for artifacts exported from Colab."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, scrolledtext

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CnnClassificationTool(ForensicsTool):
    tool_id = "cnn_classification"
    title = "CNN classification"
    category = "Lab 8"
    description = "Load a CNN exported from Colab, view its passport and classify the open image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            parent=parent, title="Choose a Lab 8 model", filetypes=[("Keras model", "*.keras")]
        )
        if not filename:
            return None
        model_path = Path(filename)
        passport_path = model_path.with_suffix(".passport.json")
        passport_text = passport_path.read_text(encoding="utf-8")
        passport = json.loads(passport_text)
        try:
            from tensorflow import keras
        except ModuleNotFoundError as error:
            raise RuntimeError("Install the dependencies from requirements.txt.") from error
        model = keras.models.load_model(model_path, compile=False)
        _, height, width, _ = model.input_shape
        image = document.current.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        probabilities = model.predict(np.asarray(image)[np.newaxis], verbose=0)[0]
        best = int(probabilities.argmax())
        prediction = passport["class_names"][best]

        window = tk.Toplevel(parent)
        window.title(f"Model passport · {passport['model']}")
        text = scrolledtext.ScrolledText(window, width=75, height=26, wrap="word")
        text.insert("1.0", passport_text)
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        return ToolResult(
            message=f"{prediction} ({probabilities[best]:.1%})",
            details={
                "Model": passport["model"],
                "Dataset": passport["dataset"],
                "Prediction": prediction,
                "Probability": f"{probabilities[best]:.1%}",
            },
        )


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection"
    category = "Lab 8"
    description = "Project saved CNN test features with t-SNE and UMAP."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        filename = filedialog.askopenfilename(
            parent=parent, title="Choose exported test features", filetypes=[("NumPy archive", "*.npz")]
        )
        if not filename:
            return None
        with np.load(filename) as saved:
            features = saved["features"]
            labels = saved["labels"]
            class_names = saved["class_names"]
        figure = draw_projections(features, labels, class_names)
        return ToolResult(
            image=figure,
            message="Projected test features with t-SNE and UMAP.",
            details={"Samples": len(features), "Feature dimensions": features.shape[1]},
        )


def draw_projections(features: np.ndarray, labels: np.ndarray, class_names: np.ndarray) -> Image.Image:
    try:
        from matplotlib.figure import Figure
        from umap import UMAP
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the dependencies from requirements.txt.") from error
    if features.ndim != 2 or len(features) < 4 or labels.shape != (len(features),):
        raise ValueError("Expected at least four feature vectors with one class label per vector.")

    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), init="random", random_state=42)
    umap = UMAP(n_components=2, n_neighbors=min(15, len(features) - 1), random_state=42, n_jobs=1)
    figure = Figure(figsize=(13, 5), layout="constrained")
    axes = figure.subplots(1, 2)
    for ax, name, projection in zip(axes, ("t-SNE", "UMAP"), (tsne, umap)):
        coordinates = projection.fit_transform(features)
        for index, class_name in enumerate(class_names):
            points = coordinates[labels == index]
            ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.55, label=str(class_name))
        ax.set(title=name, xticks=[], yticks=[])
    axes[-1].legend(title="True class", loc="upper left", bbox_to_anchor=(1.02, 1))
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=140)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return image.convert("RGB")
