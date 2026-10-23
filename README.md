# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Lab 5: matching and image lookup

- **Match second image (Set 5.3):** open a second image, choose ORB or SIFT,
  maximum keypoints, and a match ratio (default 0.8). Matching points appear
  connected across the two images. Lower ratios reject more ambiguous matches.
- **Find closest image (Set 5.4):** choose a folder and the same settings.
  The tool extracts the current image's descriptors once and returns the folder
  image with the most matches. It checks supported image files directly in that
  folder, excludes the current source file, and skips unreadable images.
  Ties use the first filename in sorted order; zero matches produces no result.

Both tools reuse the Set 5.2 descriptor function, with Hamming distance for ORB,
Euclidean distance for SIFT, mutual nearest-neighbour checks, and a ratio test.
SIFT uses the helper's defaults: 8 octaves, 3 scales, and sigma 1.6.
Results reports the match count and settings. The comparison becomes the current
image, following the existing tool pattern; use **Undo** before another search
or comparison so it runs on the unmarked input. Results can be saved normally.
Folder lookup runs synchronously, so large folders can take time.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
