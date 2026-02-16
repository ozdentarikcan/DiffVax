## DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing (ICLR 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2411.17957-b31b1b.svg)](https://arxiv.org/pdf/2411.17957)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://diffvax.github.io/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/ozdentarikcan/DiffVaxDataset)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ozdentarikcan.DiffVax)

[Tarik Can Ozden](https://ozdentarikcan.github.io/)\*,
[Ozgur Kara](https://karaozgur.com/)\*,
[Oguzhan Akcin](https://scholar.google.com/citations?user=2elIEXoAAAAJ&hl=en),
[Kerem Zaman](https://keremzaman.com/),
[Shashank Srivastava](https://scholar.google.com/citations?user=-vKI5s0AAAAJ&hl=en),
[Sandeep P. Chinchali](https://scholar.google.com/citations?user=262ASa4AAAAJ&hl=en),
[James M. Rehg](https://rehg.org/)

\* Equal Contribution

![motivation](assets/diffvax_motivation.png)

## Abstract
<b>TL; DR:</b> DiffVax is a scalable, lightweight, and optimization-free image immunization framework designed to protect images and videos from diffusion-based editing.

<details><summary>Click for the full abstract</summary>


> Current image immunization defense techniques against diffusion-based editing embed imperceptible noise into target images to disrupt editing models. However, these methods face scalability challenges, as they require time-consuming optimization for each image separately, taking hours for small batches. To address these challenges, we introduce DiffVax, a scalable, lightweight, and optimization-free framework for image immunization, specifically designed to prevent diffusion-based editing. Our approach enables effective generalization to unseen content, reducing computational costs and cutting immunization time from days to milliseconds, achieving a speedup of 250,000×. This is achieved through a loss term that ensures the failure of editing attempts and the imperceptibility of the perturbations. Extensive qualitative and quantitative results demonstrate that our model is scalable, optimization-free, adaptable to various diffusion-based editing tools, robust against counter-attacks, and, for the first time, effectively protects video content from editing.


</details>

## Installation

```bash
# Clone the repository
git clone https://github.com/ozdentarikcan/DiffVax.git
cd DiffVax

# Create conda environment (recommended)
conda create -n diffvax python=3.12 -y
conda activate diffvax

# Install dependencies
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Dataset

The DiffVax dataset is hosted on Hugging Face: [`ozdentarikcan/DiffVaxDataset`](https://huggingface.co/datasets/ozdentarikcan/DiffVaxDataset)

Download it with the provided script:

```bash
python scripts/download_dataset.py
```

This places the dataset in `data/` with the following structure:

```
data/
├── train/
│   ├── images/          # Training images (512x512 PNG)
│   ├── masks/           # Corresponding masks
│   └── metadata.jsonl   # Image-prompt pairs
└── validation/
    ├── images/
    ├── masks/
    └── metadata.jsonl
```

The training and demo scripts will also auto-download the dataset on first run if it's not present.

## Project Structure

```
DiffVax/
├── app.py                              # Gradio web demo
├── src/diffvax/                        # Main package
│   ├── attack.py                       # Stable Diffusion inpainting wrapper
│   ├── model.py                        # Nested U-Net (UNet++) architecture
│   ├── utils.py                        # Image I/O, data loading, seeding
│   ├── immunization/
│   │   ├── diffvax_immunization.py     # DiffVax immunization (trained NestedUNet)
│   │   ├── photoguard_immunization.py  # PhotoGuard baseline (PGD encoder attack)
│   │   └── diffusionguard_immunization.py  # DiffusionGuard baseline (PGD noise maximization)
│   └── metrics/                        # Image quality metrics
│       ├── base.py, factory.py
│       └── psnr.py, ssim.py, fsim.py, clip_score.py
│
├── scripts/
│   ├── train.py                        # Train the immunization model
│   ├── demo.py                         # End-to-end demo with comparison output
│   ├── evaluate.py                     # Calculate image quality metrics
│   ├── compare_baselines.py            # Multi-image baseline comparison figure
│   └── download_dataset.py             # Download dataset from Hugging Face
│
├── notebooks/
│   ├── diffvax_demo.ipynb              # Interactive demo notebook
│   └── diffvax_comparison.ipynb        # Multi-method comparison notebook
│
├── configs/
│   └── train.yml                       # Training hyperparameters
│
├── checkpoints/
│   └── diffvax_trained.pth             # Pre-trained model weights
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### Web demo (Gradio)

Launch the interactive web interface:

```bash
python app.py
```

This starts a Gradio app at `http://localhost:7860` where you can upload an image and mask, enter an editing prompt, and see how DiffVax protects the image.

### Run the demo

The demo script downloads the dataset (if needed), loads the pre-trained checkpoint, immunizes a validation image, runs the same edit on both original and immunized versions, and saves a side-by-side comparison.

```bash
python scripts/demo.py
```

Options:

```bash
# Use a different validation image and prompt
python scripts/demo.py --image-index 2 --edit-prompt "a watercolor painting"

# Run on a specific GPU
CUDA_VISIBLE_DEVICES=4 python scripts/demo.py

# Headless server (no display)
python scripts/demo.py --no-display --save-dir outputs/my_demo
```

If the dataset is not found locally, the demo generates a synthetic sample image automatically. Output files are saved to `outputs/demo/` by default:
- `<name>_original.png` — the original image
- `<name>_immunized.png` — the DiffVax-protected image
- `<name>_edited_original.png` — inpainting edit on the original
- `<name>_edited_immunized.png` — inpainting edit on the immunized image (edit should be disrupted)
- `<name>_comparison.png` — 2x2 comparison grid

You can also provide your own image and mask directly:

```bash
python scripts/demo.py --image photo.png --mask mask.png --edit-prompt "a cat sitting"
```

### Baseline Comparisons

DiffVax includes implementations of two prior-work baselines for side-by-side comparison:

- **PhotoGuard** (Salman et al., ICML 2023) — PGD encoder attack that forces `VAE.encode(x+δ)` toward a target latent (1000 L∞-PGD steps, ε=0.06), plus a diffusion attack that backpropagates through the full inpainting pipeline (200 L2-PGD steps, ε=16, 10 gradient averaging reps).
- **DiffusionGuard** (Li et al., ICLR 2025) — PGD optimization that maximizes the UNet's noise prediction norm at the highest timestep, with contour-based mask augmentation for robustness (800 L∞-PGD steps, ε=16/255).

Unlike these methods, DiffVax uses a trained NestedUNet for single-pass inference (no per-image optimization).

**Generate a multi-image comparison figure:**

```bash
python scripts/compare_baselines.py
python scripts/compare_baselines.py --images 1 5 9 29 33
```

**Run the demo with all methods compared:**

```bash
python scripts/demo.py --run-baselines --no-display
```

This produces the standard DiffVax 2x2 grid plus a 1x5 baseline comparison figure:
`Original | Edited (No Defense) | Edited (PhotoGuard) | Edited (DiffusionGuard) | Edited (DiffVax)`

You can tune PGD steps:

```bash
python scripts/demo.py --run-baselines --pg-steps 200 --dg-steps 100
```

### Train a new model

The training script automatically downloads the DiffVax dataset from Hugging Face Hub on first run.

```bash
python scripts/train.py \
    --config configs/train.yml \
    --data-dir data \
    --output-dir outputs
```

## Configuration

Training hyperparameters are set in `configs/train.yml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iter_num` | 1000000 | Number of training epochs |
| `learning_rate` | 0.00001 | Adam optimizer learning rate |
| `batch_size` | 5 | Training batch size |
| `alpha` | 4 | Weight for perturbation imperceptibility loss |
| `attack_model_link` | `runwayml/stable-diffusion-inpainting` | Diffusion model to defend against |
| `train_all` | true | Train on all images (false = use `image_index_list`) |

## Architecture

**NestedUNet (UNet++)** — a hierarchical encoder-decoder with dense nested skip connections:

- **Input**: 3-channel RGB image (512 x 512)
- **Encoder**: 5 levels with filter depths [32, 64, 128, 256, 512]
- **Decoder**: dense skip connections at every level (not just symmetric pairs)
- **Output**: 3-channel perturbation map, applied additively to the input image
- **Parameters**: ~9.2M

The perturbation is applied additively only outside the edit region (masked area), and the resulting image is clamped to the valid pixel range.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU with 8 GB+ VRAM (16 GB+ recommended for training)
- ~5 GB disk space for the Stable Diffusion Inpainting model (downloaded automatically)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/diffvax_demo.ipynb` | Interactive demo: immunize an image and compare edits in a 1x3 grid |
| `notebooks/diffvax_comparison.ipynb` | Compare DiffVax against PhotoGuard and DiffusionGuard side-by-side |

## Citation 

```
@inproceedings{ozden2026diffvax,
  title={DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing},
  author={Ozden, Tarik Can and Kara, Ozgur and Akcin, Oguzhan and Zaman, Kerem and Srivastava, Shashank and Chinchali, Sandeep P and Rehg, James M},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},      
}

``` 

## License

MIT License. See [LICENSE](LICENSE) for details.
