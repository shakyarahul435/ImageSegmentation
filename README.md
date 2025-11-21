# Building Segmentation & Damage Assessment from Satellite/Aerial Imagery

A complete pipeline for:
- Downloading & preparing satellite imagery + building masks
- Training **two segmentation models** (U-Net & ResNet34-based)
- Detecting **structural damage** by comparing before/after images
- Visualizing results with overlays and damage reports

Perfect for disaster response, insurance assessment, or urban monitoring.

---

### Features

- Downloads real satellite imagery from Hugging Face (`PuTorch/DesignEarth`)
- Trains lightweight U-Net and ResNet34 segmentation models
- Compares **before** and **after** disaster images
- Automatically flags **damaged buildings** using IoU + area change
- Generates visual reports + CSV summary
- Runs inference on custom test sets

---

### Project Structure
```
├── data_seg/                  ← Downloaded images + masks
├── Test_before/               ← Your "before disaster" images 
├── Test_after/                ← Your "after disaster" images (same filenames)
├── Test/                      ← Optional: single-image inference folder
├── results_damage/            ← Damage comparison results + CSV
├── results_test/              ← Predictions on Test/ folder
└── damage_segmentation.ipynb  ← Or .py version of this code
```



---

### Requirements

- Python 3.11
- CUDA-capable GPU recommended (works on CPU too)

### Installation

```bash
# 1. Create virtual environment (recommended)
py -3.11 -m venv seg-env
source seg-env/bin/activate    # Linux/Mac
# or
seg-env\Scripts\activate      # Windows

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pillow matplotlib datasets huggingface_hub ultralytics tqdm

# CPU-only version (if no GPU):
# pip install torch torchvision torchaudio

```
### Note:

Working on data findings for Testing before and after image of damage caused by earthquake.
