"""Set 7: bag of visual words (codebook, histograms and classification)."""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
import pickle
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "data" / "Set7_images"
MODEL_NAME = "bovw_model.pkl"
N_EXAMPLES = 10  # training images shown with their keypoints
N_LINES = 60  # keypoints linked to their histogram bin
BAR_W, MARGIN, PLOT_H = 30, 40, 240  # histogram layout


def split_dataset(folder: Path) -> None:
    """Copy <folder>/<class>/*.jpg into <folder>/{train,test}/<class>/ (75 % / 25 %)."""
    for class_folder in sorted(p for p in folder.iterdir() if p.is_dir() and p.name not in ("train", "test")):
        paths = sorted(class_folder.glob("*.jpg"))
        train, test = train_test_split(paths, test_size=0.25, random_state=42)
        for split, split_paths in (("train", train), ("test", test)):
            (folder / split / class_folder.name).mkdir(parents=True, exist_ok=True)
            for path in split_paths:
                shutil.copy(path, folder / split / class_folder.name / path.name)


def sift_features(image: Image.Image, max_features: int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Up to ``max_features`` SIFT keypoints and their 128-D descriptors (empty array if none are found)."""
    gray = np.asarray(image.convert("L"))
    keypoints, descriptors = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(gray, None)
    if descriptors is None:
        return [], np.zeros((0, 128), np.float32)
    return keypoints, descriptors


def assign_words(descriptors: np.ndarray, kmeans: MiniBatchKMeans) -> tuple[np.ndarray, np.ndarray]:
    """Map each descriptor to its closest visual word and count the (normalised) word frequencies."""
    k = kmeans.n_clusters
    if len(descriptors) == 0:
        return np.zeros(0, int), np.zeros(k)
    words = kmeans.predict(descriptors)
    counts = np.bincount(words, minlength=k)
    return words, counts / counts.sum()


def word_color(word: int, k: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(word / k, 0.9, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_histogram(histogram: np.ndarray, patches: np.ndarray, title: str) -> Image.Image:
    """Bar chart of a k-dimensional visual word histogram, with each word's patch under its bar."""
    plot_w = BAR_W * len(histogram)
    width, height = plot_w + 2 * MARGIN, PLOT_H + BAR_W + 2 * MARGIN + 10
    font = ImageFont.load_default(size=12)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 10), title, fill="black", font=font, anchor="ma")
    draw.line([(MARGIN, MARGIN), (MARGIN, MARGIN + PLOT_H), (MARGIN + plot_w, MARGIN + PLOT_H)], fill="black")
    peak = max(histogram.max(), 1e-12)
    for i, value in enumerate(histogram):
        x = MARGIN + i * BAR_W
        draw.rectangle([x + 1, MARGIN + PLOT_H * (1 - value / peak), x + BAR_W - 1, MARGIN + PLOT_H],
                       fill="lightgray", outline="black")
        patch = Image.fromarray(patches[i]).resize((BAR_W - 2, BAR_W - 2))
        image.paste(patch, (x + 1, MARGIN + PLOT_H + 4))
        draw.text((x + BAR_W / 2, MARGIN + PLOT_H + BAR_W + 6), str(i), fill="black", font=font, anchor="ma")
    draw.text((MARGIN - 4, MARGIN), f"{peak:.2f}", fill="black", font=font, anchor="rm")
    draw.text((width / 2, height - 14), "Visual word", fill="black", font=font, anchor="ma")
    return image


def draw_assignments(image: Image.Image, model: dict, title: str) -> tuple[Image.Image, np.ndarray]:
    """Image next to its visual word histogram, with lines from some keypoints to the bin they fall in."""
    keypoints, descriptors = sift_features(image, model["max_features"])
    words, histogram = assign_words(descriptors, model["kmeans"])
    chart = draw_histogram(histogram, model["patches"], title)
    scale = chart.height / image.height
    photo = image.convert("L").resize((round(image.width * scale), chart.height))
    gap = 40
    canvas = Image.new("RGB", (photo.width + gap + chart.width, chart.height), "white")
    canvas.paste(photo, (0, 0))
    canvas.paste(chart, (photo.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    k, peak = len(histogram), max(histogram.max(), 1e-12)
    shown = np.random.default_rng(0).choice(len(keypoints), min(N_LINES, len(keypoints)), replace=False)
    points = [(keypoints[i].pt[0] * scale, keypoints[i].pt[1] * scale, words[i]) for i in shown]
    for x, y, word in points:
        bar_x = photo.width + gap + MARGIN + (word + 0.5) * BAR_W
        bar_y = MARGIN + PLOT_H * (1 - histogram[word] / peak)
        draw.line([(x, y), (bar_x, bar_y)], fill=(160, 160, 160))
    for x, y, word in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=word_color(word, k), outline="black")
    return canvas, histogram


def load_model(parent: tk.Misc) -> dict | None:
    folder = filedialog.askdirectory(
        title="Choose dataset folder (with the saved codebook)", initialdir=DEFAULT_FOLDER, parent=parent
    )
    if not folder:
        return None
    model_path = Path(folder) / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"No {MODEL_NAME} in {folder}. Run 'Codebook construction' first.")
    with open(model_path, "rb") as file:
        return pickle.load(file)


class CodebookTool(ForensicsTool):
    tool_id = "codebook"
    title = "Codebook construction"
    category = "Lab 7"
    description = "Cluster SIFT descriptors of the training images into K visual words (default 10 x classes)."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        folder = filedialog.askdirectory(title="Choose dataset folder", initialdir=DEFAULT_FOLDER, parent=parent)
        if not folder:
            return None
        folder = Path(folder)
        if not (folder / "train").is_dir():
            split_dataset(folder)
        classes = sorted(p.name for p in (folder / "train").iterdir() if p.is_dir())
        k = simpledialog.askinteger(
            "Codebook construction", "Number of visual words K:", initialvalue=10 * len(classes), minvalue=2,
            parent=parent,
        )
        if k is None:
            return None
        max_features = simpledialog.askinteger(
            "Codebook construction", "Maximum SIFT features per image:", initialvalue=500, minvalue=1, parent=parent
        )
        if max_features is None:
            return None

        # SIFT descriptors of every training image
        paths = sorted((folder / "train").glob("*/*.jpg"))
        labels, images, features = [], [], []
        for path in paths:
            image = Image.open(path).convert("L")
            labels.append(path.parent.name)
            images.append(image)
            features.append(sift_features(image, max_features))
        all_descriptors = np.vstack([descriptors for _, descriptors in features])

        # K-means: each cluster centre is a visual word
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42).fit(all_descriptors)

        # Train the classifier on the training histograms
        histograms = np.array([assign_words(descriptors, kmeans)[1] for _, descriptors in features])
        classifier = SVC(kernel="rbf", probability=True, random_state=42).fit(histograms, labels)

        # A centre is not an image: show each word by the training patch whose descriptor is closest to it
        owners = [(i, keypoint) for i, (keypoints, _) in enumerate(features) for keypoint in keypoints]
        closest = kmeans.transform(all_descriptors).argmin(axis=0)  # (k,)
        tile, label_h, gap = 48, 14, 6
        patches = []
        for idx in closest:
            image_idx, keypoint = owners[idx]
            (x, y), r = keypoint.pt, max(3 * keypoint.size, 8)  # region covered by the SIFT descriptor
            patches.append(np.asarray(images[image_idx].crop((x - r, y - r, x + r, y + r)).resize((tile, tile))))
        patches = np.array(patches)

        model_path = folder / MODEL_NAME
        with open(model_path, "wb") as file:
            pickle.dump({"kmeans": kmeans, "classifier": classifier, "patches": patches,
                         "max_features": max_features}, file)

        # Top: some training images with their keypoints in red. Bottom: the K visual words.
        thumb_w, thumb_h = 140, 105
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k / cols)
        width = max(5 * (thumb_w + gap), cols * (tile + gap)) + gap
        words_y = 2 * (thumb_h + gap) + 2 * gap
        result = Image.new("RGB", (width, words_y + rows * (tile + label_h + gap)), "white")
        draw = ImageDraw.Draw(result)
        font = ImageFont.load_default(size=11)
        for n, i in enumerate(np.random.default_rng(42).choice(len(images), N_EXAMPLES, replace=False)):
            scale = min(thumb_w / images[i].width, thumb_h / images[i].height)
            x0, y0 = gap + (n % 5) * (thumb_w + gap), gap + (n // 5) * (thumb_h + gap)
            result.paste(images[i].resize((round(images[i].width * scale), round(images[i].height * scale))), (x0, y0))
            for keypoint in features[i][0]:
                x, y = x0 + keypoint.pt[0] * scale, y0 + keypoint.pt[1] * scale
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="red")
        for word, patch in enumerate(patches):
            row, col = divmod(word, cols)
            x0, y0 = gap + col * (tile + gap), words_y + row * (tile + label_h + gap)
            draw.text((x0 + tile / 2, y0), f"vw {word}", fill="black", font=font, anchor="ma")
            result.paste(Image.fromarray(patch), (x0, y0 + label_h))

        return ToolResult(
            image=result,
            message=f"Top: training images with their SIFT keypoints. Bottom: the {k} visual words "
                    f"(closest training patch to each centre). Model saved to {model_path.name}.",
            details={
                "Operation": "Codebook construction",
                "Classes": ", ".join(classes),
                "Train images": len(paths),
                "Descriptors": len(all_descriptors),
                "Visual words K": k,
                "Max features per image": max_features,
                "Model": str(model_path),
            },
        )


class VisualWordHistogramTool(ForensicsTool):
    tool_id = "visual_word_histogram"
    title = "Histogram construction"
    category = "Lab 7"
    description = "Assign the SIFT features of a test image to visual words and show the k-bin histogram."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        if document.path is None or document.path.parent.parent.name != "test":
            return ToolResult(message="Open a test image (from the test/<class>/ folder) first.")
        model = load_model(parent)
        if model is None:
            return None
        figure, histogram = draw_assignments(
            document.current, model, f"Histogram of visual words ({document.path.parent.name})"
        )
        return ToolResult(
            image=figure,
            message=f"Histogram over {len(histogram)} visual words; lines link {N_LINES} keypoints to their bin.",
            details={
                "Operation": "Histogram construction",
                "Visual words K": len(histogram),
                "Most frequent word": int(histogram.argmax()),
            },
        )


class VisualWordClassificationTool(ForensicsTool):
    tool_id = "visual_word_classification"
    title = "Classification"
    category = "Lab 7"
    description = "Show the visual word histogram of the image and classify it with the trained SVM."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        model = load_model(parent)
        if model is None:
            return None
        figure, histogram = draw_assignments(document.current, model, "Histogram of visual words")
        classifier = model["classifier"]

        window = tk.Toplevel(parent)
        window.title("Classification")
        photo = ImageTk.PhotoImage(figure)
        label = ttk.Label(window, image=photo)
        label.image = photo  # keep a reference so Tk does not drop the image
        label.pack(padx=10, pady=10)
        result = ttk.Label(window, text="", font=("", 14))

        def classify() -> None:
            probabilities = classifier.predict_proba([histogram])[0]
            best = probabilities.argmax()
            result.config(text=f"p({classifier.classes_[best]}) = {probabilities[best]:.2f}")

        ttk.Button(window, text="Classify", command=classify).pack(pady=(0, 5))
        result.pack(pady=(0, 10))
        return ToolResult(
            message="Opened the visual word histogram. Click 'Classify' to predict the class.",
            details={"Operation": "Classification", "Visual words K": len(histogram)},
        )
