# Document Scanner

A lightweight command-line document scanner inspired by applications such as CamScanner. This project uses classical computer vision techniques to automatically detect document boundaries, correct perspective distortion, and enhance scanned documents.

Built with **OpenCV** and **NumPy**.

---

## Features

- Terminal file browser for picking a source image (no path needed)
- Automatic document boundary detection
- Interactive corner adjustment
- Perspective correction
- Document Postprocessing with White Balance, Illumination Correction.

---

## Demo

| Original | Detected | Scanned |
|----------|----------|----------|
| ![Original](/docs/showcase/original.jpg) | ![Detected](/docs/showcase/detect.png)| ![Scanned](/docs/showcase/scan.jpg) |

---

## Processing Pipeline

The scanner follows the pipeline below:

1. Load an image
2. Detect document edges
3. Approximate the document contour
4. Allow manual corner adjustment
5. Apply perspective transformation
6. Enhance the document
7. Save the scanned image

```
Image
  │
  ▼
Document Detection
  │
  ▼
Corner Selection
  │
  ▼
Perspective Warp
  │
  ▼
Image Enhancement
  │
  ▼
Scanned Document
```

---

## Tech Stack

- OpenCV
- NumPy
- Pillow

---

## Installation

### Clone the repository

```bash
git clone https://github.com/manhasv/scanner.git
cd scanner
```

### Option 1: Using Python virtual environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

### Option 2: Using Conda

Install using yml:

```bash
conda env create -f environment.yml
```
or 

Create a new environment:

```bash
conda create -n scanner python=3.11
```

Activate the environment:

```bash
conda activate scanner
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

### Run the application

```bash
python cli.py path/to/photo.jpg -o scan.pdf
```

Omit the image argument to pick a file with a terminal file browser (arrow keys/j-k to navigate, Enter to open a folder or select a file, Backspace/h to go up a directory, q to cancel), and omit `-o/--output` to save alongside the input as `<input>_scan.jpg`. Detected corners open in an adjustable window (drag corners, Enter to confirm, Esc to cancel) before the scan is warped and saved. Run `python cli.py --help` for all options.

---

## Current Image Enhancement

The current enhancement pipeline includes:

- Gray World white balancing
- Perspective correction

Several additional enhancement methods were explored during development, including CLAHE, homomorphic filtering, and illumination correction. These methods are discussed in the project report.

---

## Limitations

Current limitations include:

- Assumes the largest quadrilateral is the document.
- Performance may degrade under severe shadows or cluttered backgrounds.
- Extremely curved or folded documents are not handled well.

---

## License

This project is provided for educational purposes.
