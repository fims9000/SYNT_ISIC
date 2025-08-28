## ISIC Synthetic Data Generator (GUI) with Explainable AI

This repository provides a self-contained graphical application for generating synthetic dermatological images using diffusion models (UNet + DDPM scheduler) and for performing per-image explainability (XAI). The application is engineered to be cloned and executed from any directory without path-dependent assumptions; all runtime paths are resolved relative to the project root.

The training code under `diffusion/` is included for reference and scientific completeness; it is not required to operate the GUI.

### Tracked repository contents

```
ISICGUI/
├── core/                  # Core library: config, IO paths, logging, caching, generation
├── diffusion/             # Reference training code (not required for GUI runtime)
├── xai/                   # XAI integration used by the GUI
├── download_models.py     # Utility to fetch pretrained checkpoints
├── main.py                # PyQt5 GUI entry point
├── run_isicgui.bat        # Windows launcher (double-click to start GUI)
└── requirements.txt       # Python dependencies
```

Note: Runtime artifacts (model files, generated images, logs, caches, XAI outputs, etc.) are deliberately not tracked and will be created locally when you run the application.

## Installation

Prerequisites:
- Python 3.8–3.11
- Optional but recommended: NVIDIA GPU with CUDA for faster generation (CPU mode is supported)

Steps:
```bash
git clone https://github.com/<your-org-or-user>/ISICGUI.git
cd ISICGUI

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Model Weights (download_models.py)

The GUI expects pretrained diffusion UNet checkpoints per ISIC class and an auxiliary classifier for XAI. Use the downloader to populate the local model directory:

```bash
python download_models.py
```

Behavior and guarantees:
- Downloads pre-trained diffusion checkpoints and metadata into a local `checkpoints` directory under the project root.
- The script is idempotent: partial downloads can be re-run; previously downloaded files are reused or verified.
- On Windows, `download_models.bat` may be used equivalently if present.

## Launching the GUI

- Windows: double-click `run_isicgui.bat` (preferred) or run `python main.py` from any directory.
- macOS/Linux/WSL: run `python main.py` in the repository root with the virtual environment activated.

Architectural note: all paths are computed relative to the project root (the folder containing `main.py`). The application does not rely on the current working directory.

## Functional Overview

- Per-class synthetic generation using UNet2D with attention at intermediate depths and a DDPM scheduler.
- Reproducibility via a global base seed (default 42) plus deterministic per-class offsets and per-image indices.
- Automatic creation of per-class subdirectories under the selected output folder; filenames follow ISIC naming (`ISIC_XXXXXXX.png`) with strictly monotone numbering per class folder.
- Metadata CSV (`synthetic_dataset.csv`) at the output root recording filename, class, ISIC number, source label, and generation timestamp.
- Optional XAI overlays based on an auxiliary classifier combining Integrated Gradients and a sampling-based SHAP approximation.
- Memory-conscious generation: GPU cache cleared after each image; device memory status displayed periodically.

## User Interface and Operation

### Top controls
- Select Models: choose a folder named `checkpoints` that contains the downloaded diffusion checkpoints.
- Select Output: select the destination directory for generated images; class subfolders are created automatically.
- XAI Mode: toggle overlays on/off for the center preview. When enabled, the currently displayed image is augmented with an interpretability heatmap.
- XAI steps: integer spin control to the right of the XAI Mode button; defines the timestep stride for saving XAI steps in the full XAI pipeline (exported via `XAI_SAVE_EVERY_N`).
- Device: choose CPU or a specific CUDA device (e.g., CUDA:0). The bottom status labels include GPU memory usage, updated roughly every 2 seconds.

### Class configuration (left panel)
- Classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC.
- Availability: a class is enabled if its checkpoint `unet_<CLASS>_best.pth` is present in the selected models directory.
- Quantity: select the number of images to generate per enabled class.

### Preview and progress (center panel)
- Live image preview with proportional scaling.
- Progress bar indicates global completion. Textual logs additionally report denoising progress every few steps (i/N for the current image) and the overall count.

### Project structure and browsing (right panel)
- Logical tree with nodes for generated images, XAI results, and checkpoints.
- Two lists: class folders and image files. Clicking an image updates the center preview. Clicking the preview opens the image in the system viewer.

### Logs and configuration (bottom panel)
- Console with informational, warning, and error messages, including generation progress and XAI events.
- Static labels: device, model path, available model count, color statistics status, memory usage.

## Explainability (XAI) Details

- Periodic overlays: for each class, a lightweight XAI overlay is produced every N-th image, with N=10 by default. If fewer than N images are generated for a class, the first image receives an overlay.
- Methods: Integrated Gradients (primary) and a sampling-based SHAP approximation are combined; attributions are normalized and overlaid as a heatmap blended with the image.
- Full XAI pipeline: the GUI enqueues a complete analysis for classes at defined intervals. Subprocess execution uses UTF‑8 and a non-interactive plotting backend for robustness. The “XAI steps” spin control sets `XAI_SAVE_EVERY_N` for stepwise exports.
- Classifier: a ResNet‑18 backbone adapted to the number of classes is used if `classifier.pth` is available; otherwise, ImageNet weights are used with permissive head loading.

## How Generation Works (Algorithmic Summary)

- Model: UNet2D (sample_size=128, RGB in/out, attention blocks) with a DDPMScheduler (num_train_timesteps=1000, squaredcos_cap_v2 schedule). Inference timesteps default to 50 and are configurable.
- Seeds: a global base seed is combined with an MD5-derived class offset and the within-class index to ensure deterministic but diverse outputs across classes.
- Denoising: at each scheduler timestep, the model predicts noise; the scheduler updates the latent accordingly. Tensors and generators are placed on the selected device.
- Postprocessing: if `color_statistics.json` exists alongside checkpoints, per-class color mean/std adjustments are applied with bounded scaling and smooth blending to avoid artifacts.

## Practical Usage

1. Install dependencies and activate the environment.
2. Run `python download_models.py` to fetch models.
3. Start the GUI (`run_isicgui.bat` on Windows or `python main.py`).
4. Click “Select Models” and choose the `checkpoints` folder (created in step 2).
5. Click “Select Output” and choose an empty or existing directory for generated images.
6. Select device (CPU or CUDA). Optionally enable XAI Mode and set the “XAI steps” spin value.
7. Choose classes and quantities; click “Start”. Use “Stop” to interrupt or “Regenerate” to repeat the last configuration.
8. Inspect generated images and XAI overlays in the UI; open folders or images directly from the interface.

## Configuration and Paths

- Paths are resolved relative to the project root (directory of `main.py`), not the current working directory. The program can be launched from any folder.
- Required directories are created on demand. Logs and caches are kept local to the project.

## Troubleshooting

- GPU memory: reduce per-class quantities or switch to CPU if “CUDA out of memory” occurs.
- Slow overlays: SHAP is approximate yet heavier than IG; expect extra latency when overlays are generated.
- Rendering: the XAI subprocess uses a headless backend; images are saved even if an interactive display is unavailable.

## Notes on diffusion/

The `diffusion/` directory contains training and demonstration scripts. These files are provided for reference and replicability but are not required to run the GUI. No modifications to this folder are necessary for typical end users.

## Citation

If this software supports your research, please provide an appropriate citation acknowledging diffusion-based generation and attribution-based explainability.



