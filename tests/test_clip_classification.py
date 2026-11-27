import unittest

import numpy as np

from forensics_app.tools.clip_classification import WEBCAM_PRESETS, build_prompts, draw_overlay, probability_details


class ClipHelperTests(unittest.TestCase):
    def test_template_receives_each_class_text(self) -> None:
        self.assertEqual(
            build_prompts("a photo of {}.", ["one person", "two people"]),
            ["a photo of one person.", "a photo of two people."],
        )
        self.assertEqual(build_prompts("{}", ["a screenshot"]), ["a screenshot"])

    def test_default_webcam_classes_are_classroom_friendly(self) -> None:
        _, labels = next(iter(WEBCAM_PRESETS.values()))
        self.assertEqual(labels, ["one person", "two people", "three people", "a person holding a pen"])

    def test_details_are_sorted_by_probability(self) -> None:
        details = probability_details(["cat", "dog", "owl"], np.array([0.2, 0.7, 0.1]))
        self.assertEqual(list(details), ["  dog", "  cat", "  owl"])
        self.assertEqual(details["  dog"], "70.0%")

    def test_overlay_keeps_frame_size_and_does_not_mutate_input(self) -> None:
        frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        result = draw_overlay(frame, ["one person", "two people"], np.array([0.8, 0.2]))
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue((frame == 200).all())


if __name__ == "__main__":
    unittest.main()
