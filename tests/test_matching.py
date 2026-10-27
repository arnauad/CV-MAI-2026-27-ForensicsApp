from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.image_lookup import ImageLookupTool
from forensics_app.tools.matching import (
    ImageMatchingTool, extract_features, match_features,
)


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(42).integers(
            0, 256, (128, 128, 3), dtype=np.uint8,
        ))
        self.document = ImageDocument()
        self.document.current = self.image.copy()

    def test_both_methods_match_translated_image(self):
        shifted = Image.fromarray(np.roll(np.asarray(self.image), (7, 9), axis=(0, 1)))
        for method in ("orb", "sift"):
            with self.subTest(method=method):
                first = extract_features(self.image, method, 100)
                second = extract_features(shifted, method, 100)
                matches = match_features(first, second, method, 0.8)
                self.assertGreater(len(matches), 5)
                offsets = second[0][matches[:, 1]] - first[0][matches[:, 0]]
                np.testing.assert_allclose(np.median(offsets, axis=0), [7, 9], atol=1)

    def test_ransac_filters_outlier_matches(self):
        first = extract_features(self.image, "orb", 100)
        shifted = Image.fromarray(np.roll(np.asarray(self.image), (7, 9), axis=(0, 1)))
        second = extract_features(shifted, "orb", 100)
        matches = match_features(first, second, "orb", 0.8)
        filtered = match_features(first, second, "orb", 0.8, use_ransac=True)
        self.assertLessEqual(len(filtered), len(matches))
        self.assertGreater(len(filtered), 0)

    def test_no_features_and_too_few_descriptors(self):
        for method in ("orb", "sift"):
            for size in ((1, 1), (64, 64)):
                self.assertIsNone(extract_features(Image.new("RGB", size), method, 100))
        self.assertEqual(match_features(None, None, "orb", 0.8).shape, (0, 2))
        single = (np.zeros((1, 2)), np.zeros((1, 128)))
        self.assertEqual(len(match_features(single, single, "sift", 0.8)), 0)

    def test_ratio_rejects_ambiguous_match(self):
        first = (np.zeros((1, 2)), np.array([[0., 0.]]))
        second = (np.zeros((2, 2)), np.array([[1., 0.], [1.01, 0.]]))
        self.assertEqual(len(match_features(first, second, "sift", 0.8)), 0)
        self.assertEqual(len(match_features(first, second, "sift", 1)), 1)

    def test_second_image_output_preserves_input_and_supports_undo(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "second.png"
            self.image.save(path)
            with patch("forensics_app.tools.matching.filedialog.askopenfilename", return_value=str(path)), \
                 patch("forensics_app.tools.matching.ask_matching_settings", return_value=("orb", 100, 0.8, False)):
                result = ImageMatchingTool().run(None, self.document)
        self.assertGreater(result.details["Matches"], 5)
        self.assertEqual(result.image.size, (256, 128))
        np.testing.assert_array_equal(self.document.current, self.image)
        self.document.apply(result.image)
        self.assertTrue(self.document.undo())
        np.testing.assert_array_equal(self.document.current, self.image)

    def test_lookup_selects_best_and_skips_source_and_unreadable_images(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.document.path = root / "source.png"
            self.image.save(self.document.path)
            self.image.save(root / "best.PNG")
            Image.new("RGB", (128, 128)).save(root / "blank.png")
            Image.fromarray(np.random.default_rng(9).integers(
                0, 256, (128, 128, 3), dtype=np.uint8,
            )).save(root / "other.png")
            (root / "broken.png").write_text("not an image")
            with patch("forensics_app.tools.image_lookup.filedialog.askdirectory", return_value=folder), \
                 patch("forensics_app.tools.image_lookup.ask_matching_settings", return_value=("orb", 100, 0.8, False)):
                result = ImageLookupTool().run(None, self.document)
            self.assertEqual(Path(result.details["Closest image"]).name, "best.PNG")
            self.assertEqual(result.details["Images checked"], 3)
            self.assertEqual(result.details["Unreadable images"], 1)
            self.assertEqual(result.details["Images without keypoints"], 1)
            self.assertGreater(result.details["Matches"], 5)
            np.testing.assert_array_equal(self.document.current, self.image)

    def test_empty_folder_and_zero_matches_do_not_select_an_image(self):
        with TemporaryDirectory() as folder:
            for blank in (False, True):
                if blank:
                    Image.new("RGB", (128, 128)).save(Path(folder) / "blank.png")
                with patch("forensics_app.tools.image_lookup.filedialog.askdirectory", return_value=folder), \
                     patch("forensics_app.tools.image_lookup.ask_matching_settings", return_value=("orb", 100, 0.8, False)):
                    result = ImageLookupTool().run(None, self.document)
                self.assertIsNone(result.image)
                self.assertNotIn("Closest image", result.details)

    def test_cancel_is_harmless(self):
        with patch("forensics_app.tools.matching.filedialog.askopenfilename", return_value=""):
            self.assertIsNone(ImageMatchingTool().run(None, self.document))
        with patch("forensics_app.tools.image_lookup.filedialog.askdirectory", return_value=""):
            self.assertIsNone(ImageLookupTool().run(None, self.document))

    def test_tools_are_registered_in_lab5(self):
        categories = dict(build_tool_registry().categories())
        ids = {tool.tool_id for tool in categories["Lab 5"]}
        self.assertTrue({"image_matching", "image_lookup"} <= ids)


if __name__ == "__main__":
    unittest.main()
