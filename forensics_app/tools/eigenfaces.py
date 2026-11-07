"""Set 6.2: eigenfaces (PCA) on LFW and nearest-neighbour face recognition."""

from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "data" / "Set6_materials" / "lfw"
MODEL_NAME = "eigenfaces_model.npz"


def save_lfw_dataset(folder: Path) -> None:
    """Download LFW and save it as <folder>/{train,test}/<Person_Name>/<index>.png."""
    lfw_people = fetch_lfw_people(data_home=str(folder), min_faces_per_person=70, resize=0.4)
    images = (lfw_people.images * 255).astype(np.uint8)  # LFW pixels are floats in [0, 1]
    names = lfw_people.target_names[lfw_people.target]
    train_test = train_test_split(images, names, test_size=0.25, random_state=42)
    for split, split_images, split_names in (("train", train_test[0], train_test[2]),
                                             ("test", train_test[1], train_test[3])):
        for i, (image, name) in enumerate(zip(split_images, split_names)):
            person_folder = folder / split / name.replace(" ", "_")
            person_folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(person_folder / f"{i:04d}.png")


def load_split(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened images in [0, 1] and their person names (taken from the folder names)."""
    paths = sorted(folder.glob("*/*.png"))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}.")
    images = [np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255 for path in paths]
    names = [path.parent.name.replace("_", " ") for path in paths]
    return np.array([image.ravel() for image in images]), np.array(names)


def to_uint8(values: np.ndarray, shape: tuple[int, int]) -> Image.Image:
    """Min-max scale an array (e.g. an eigenface) to a displayable grayscale image."""
    values = values.reshape(shape)
    values = (values - values.min()) / max(np.ptp(values), 1e-12)
    return Image.fromarray((values * 255).astype(np.uint8))


class EigenfacesTool(ForensicsTool):
    tool_id = "eigenfaces"
    title = "Eigenfaces (PCA)"
    category = "Lab 6"
    description = "Save LFW faces to train/test folders, compute the eigenfaces and save the model."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        folder = filedialog.askdirectory(
            title="Choose face dataset folder", initialdir=DEFAULT_FOLDER.parent, parent=parent
        )
        if not folder:
            return None
        n_components = simpledialog.askinteger(
            "Eigenfaces", "Number of eigenfaces K:", initialvalue=100, minvalue=1, parent=parent
        )
        if n_components is None:
            return None

        folder = Path(folder)
        if not (folder / "train").is_dir():
            save_lfw_dataset(folder)
        X_train, names = load_split(folder / "train")
        h, w = np.asarray(Image.open(next((folder / "train").glob("*/*.png")))).shape
        n_components = min(n_components, *X_train.shape)

        # Mean, eigenfaces U_K and PCA coefficients w_i = U_K^T (x_i - mu)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42).fit(X_train)
        mean = pca.mean_
        eigenfaces = pca.components_
        coefficients = (X_train - mean) @ eigenfaces.T
        model_path = folder / MODEL_NAME
        np.savez(model_path, mean=mean, eigenfaces=eigenfaces, coefficients=coefficients,
                 names=names, shape=np.array([h, w]))

        # Gallery with the first K eigenfaces
        scale, gap = 2, 4
        cols = math.ceil(math.sqrt(n_components))
        rows = math.ceil(n_components / cols)
        tile_w, tile_h = w * scale, h * scale
        gallery = Image.new("L", (cols * (tile_w + gap) + gap, rows * (tile_h + gap) + gap), 255)
        for k, eigenface in enumerate(eigenfaces):
            tile = to_uint8(eigenface, (h, w)).resize((tile_w, tile_h), Image.Resampling.BICUBIC)
            row, col = divmod(k, cols)
            gallery.paste(tile, (gap + col * (tile_w + gap), gap + row * (tile_h + gap)))

        return ToolResult(
            image=gallery,
            message=f"First {n_components} eigenfaces (left to right, top to bottom). Model saved to {model_path.name}.",
            details={
                "Operation": "Eigenfaces (PCA)",
                "Train images": len(X_train),
                "People": len(np.unique(names)),
                "Face size (h x w)": f"{h} x {w}",
                "Eigenfaces K": n_components,
                "Explained variance": f"{100 * pca.explained_variance_ratio_.sum():.1f} %",
                "Model": str(model_path),
            },
        )


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face Recognition (Eigenfaces)"
    category = "Lab 6"
    description = "Find the closest training faces to the current test face in eigenface space."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        folder = filedialog.askdirectory(
            title="Choose face dataset folder (with the saved model)", initialdir=DEFAULT_FOLDER, parent=parent
        )
        if not folder:
            return None
        n_matches = simpledialog.askinteger(
            "Face Recognition", "Number of closest faces to show:", initialvalue=3, minvalue=1, maxvalue=10,
            parent=parent,
        )
        if n_matches is None:
            return None

        folder = Path(folder)
        model_path = folder / MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"No {MODEL_NAME} in {folder}. Run 'Eigenfaces (PCA)' first.")
        model = np.load(model_path)
        mean, eigenfaces, coefficients, names = model["mean"], model["eigenfaces"], model["coefficients"], model["names"]
        h, w = model["shape"]
        X_train, _ = load_split(folder / "train")

        # w_test = U_K^T (x_test - mu), then d_i = ||w_test - w_i||_2
        test = document.current.convert("L").resize((int(w), int(h)), Image.Resampling.BICUBIC)
        x_test = np.asarray(test, dtype=np.float64).ravel() / 255
        w_test = eigenfaces @ (x_test - mean)
        distances = np.linalg.norm(coefficients - w_test, axis=1)
        closest = np.argsort(distances)[:n_matches]

        # Test face followed by its closest training faces, each with a caption
        scale, gap, caption = 4, 8, 40
        tile_w, tile_h = int(w) * scale, int(h) * scale
        font = ImageFont.load_default(size=14)
        panel = Image.new("RGB", ((n_matches + 1) * (tile_w + gap) + gap, tile_h + caption + 2 * gap), "white")
        draw = ImageDraw.Draw(panel)
        # Images saved by the Eigenfaces tool live in test/<Person_Name>/, so the folder gives the true name
        in_test = document.path is not None and document.path.parent.parent.name == "test"
        true_name = document.path.parent.name.replace("_", " ") if in_test else "Unknown"
        tiles = [(test, "Test image", true_name, "black")]
        for rank, i in enumerate(closest):
            color = "black" if not in_test else "green" if names[i] == true_name else "red"
            tiles.append((to_uint8(X_train[i], (h, w)), f"#{rank + 1}  d = {distances[i]:.2f}", names[i], color))
        for k, (face, line1, line2, color) in enumerate(tiles):
            x = gap + k * (tile_w + gap)
            panel.paste(face.resize((tile_w, tile_h), Image.Resampling.BICUBIC), (x, gap))
            draw.text((x + tile_w / 2, tile_h + gap + 4), line1, fill="black", font=font, anchor="ma")
            draw.text((x + tile_w / 2, tile_h + gap + 22), line2, fill=color, font=font, anchor="ma")

        details: dict[str, object] = {
            "Operation": "Face Recognition (Eigenfaces)",
            "Eigenfaces K": len(eigenfaces),
            "Train images": len(coefficients),
            "Prediction": names[closest[0]],
        }
        for rank, i in enumerate(closest):
            details[f"#{rank + 1}"] = f"{names[i]} (d = {distances[i]:.2f})"
        return ToolResult(
            image=panel,
            message=f"Closest training face: {names[closest[0]]} (d = {distances[closest[0]]:.2f}).",
            details=details,
        )
