import tempfile
import unittest
from pathlib import Path

from forensics_app.tools.yolo_detection import read_ground_truth


class GroundTruthTests(unittest.TestCase):
    def test_yolo_label_becomes_pixel_box(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "labels").mkdir()
            (Path(folder) / "labels" / "cell.txt").write_text("2 0.5 0.25 0.2 0.1\n")
            boxes = read_ground_truth(Path(folder) / "images" / "cell.jpg", 640, 400)
        self.assertEqual(len(boxes), 1)
        cls, box = boxes[0]
        self.assertEqual(cls, 2)
        for value, expected in zip(box, (256, 80, 384, 120)):
            self.assertAlmostEqual(value, expected)

    def test_image_without_label_file_has_no_ground_truth(self) -> None:
        self.assertEqual(read_ground_truth(Path("photo.jpg"), 100, 100), [])
        self.assertEqual(read_ground_truth(None, 100, 100), [])


if __name__ == "__main__":
    unittest.main()
