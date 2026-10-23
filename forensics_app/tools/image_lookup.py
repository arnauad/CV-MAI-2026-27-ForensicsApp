"""Set 5.4: find the folder image with the most descriptor matches."""

from pathlib import Path
from tkinter import filedialog

from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .matching import ask_matching_settings, draw_matches, extract_features, match_features


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Find closest image"
    category = "Lab 5"
    description = "Find the image in a folder with the most ORB or SIFT matches."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        folder = filedialog.askdirectory(title="Choose image folder", parent=parent)
        if not folder:
            return None
        settings = ask_matching_settings(parent, self.title)
        if settings is None:
            return None
        method, count, ratio = settings
        query = extract_features(document.current, method, count)
        if query is None:
            return ToolResult(message="No keypoints found in the current image. Try another image or detector.")

        source = document.path.resolve() if document.path is not None else None
        candidates = sorted(
            (path for path in Path(folder).iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
             and path.resolve() != source),
            key=lambda path: path.name,
        )
        checked, unreadable, no_features = 0, 0, 0
        best_path, best_image, best_features, best_matches = None, None, None, None
        for path in candidates:
            try:
                with Image.open(path) as image:
                    candidate_image = image.convert("RGB")
            except (OSError, UnidentifiedImageError):
                unreadable += 1
                continue
            checked += 1
            candidate = extract_features(candidate_image, method, count)
            if candidate is None:
                no_features += 1
                continue
            matches = match_features(query, candidate, method, ratio)
            # Keep the first filename in a tie; zero matches is not a result.
            if len(matches) > 0 and (best_matches is None or len(matches) > len(best_matches)):
                best_path, best_image = path, candidate_image
                best_features, best_matches = candidate, matches

        details = {
            "Operation": "Image lookup", "Folder": folder, "Method": method.upper(),
            "Maximum keypoints": count, "Match ratio": ratio,
            "Images checked": checked, "Unreadable images": unreadable,
            "Images without keypoints": no_features,
        }
        if best_path is None:
            return ToolResult(message="No matching image found in the selected folder.", details=details)
        details.update({"Closest image": str(best_path), "Matches": len(best_matches)})
        return ToolResult(
            image=draw_matches(document.current, best_image, query[0], best_features[0], best_matches),
            message=f"Closest image: {best_path.name} ({len(best_matches)} matches).",
            details=details,
        )
