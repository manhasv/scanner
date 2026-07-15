# Document Scanner

A lightweight web-based document scanner inspired by applications such as CamScanner. This project uses classical computer vision techniques to automatically detect document boundaries, correct perspective distortion, and enhance scanned documents through a simple web interface.

Built with **FastAPI**, **OpenCV**, and **JavaScript**.

---

## Features

- Upload images from desktop or mobile devices
- Automatic document boundary detection
- Interactive corner adjustment
- Perspective correction
- Document Postprocessing with White Balance, Illumination Correction.

---

## Demo

| Original | Detected | Scanned |
|----------|----------|----------|
| *(Add screenshot)* | *(Add screenshot)* | *(Add screenshot)* |

---

## Processing Pipeline

The scanner follows the pipeline below:

1. Upload an image
2. Detect document edges
3. Approximate the document contour
4. Allow manual corner adjustment
5. Apply perspective transformation
6. Enhance the document
7. Return the scanned image

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

## Techn Stacks

### Backend

- FastAPI
- OpenCV
- NumPy
- Pillow

### Frontend

- HTML
- CSS
- JavaScript
- HTML Canvas

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

### Run local

```bash
uvicorn src.api:app --reload
```

Then open your browser and visit:

```
http://127.0.0.1:8000
```
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
- Extremely curved or folded documents are not handled.
- OCR and PDF export are not yet implemented.

---

## Future Improvements

- OCR integration
- PDF export
- Shadow removal
- Adaptive thresholding
- Machine learning–based document detection
- Batch document scanning

---

## License

This project is provided for educational purposes.
