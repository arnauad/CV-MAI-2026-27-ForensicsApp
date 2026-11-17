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

## Labs 7 and 8

- **Lab 7:** In the desktop app, run **Codebook construction** on a dataset
  folder with one subfolder of `.jpg` images per class. The tool creates
  `train/` and `test/` splits if needed and defaults to ten visual words per
  class. Open an image from `test/<class>/` for **Histogram construction**.
  Use **Classification** on an open image,
  select the same dataset folder, then click **Classify** to see the top
  probability.
- **Lab 8:** Upload the adapted [course notebook](notebooks/image_classification_using_cnn.ipynb)
  to Google Colab and run it with a GPU runtime. The notebook trains both models on
  CIFAR-10 and packages the models, weights, passports, test features, and
  projection figure for download. The quick-start install includes the
  dependencies for the desktop Lab 8 tools. Extract the Colab archive.
  Open an image and select its classifier’s
  `.keras` file in **CNN classification**, or select the exported
  `.test_features.npz` in **Feature projection**.

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
