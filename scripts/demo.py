#!/usr/bin/env python3
"""
DiffVax Demo — End-to-end immunization and comparison.

Loads the pre-trained checkpoint, immunizes a validation image, and compares
editing results on the original vs. immunized image side-by-side.

Usage:
    python scripts/demo.py --image path/to/image.png --mask path/to/mask.png
    python scripts/demo.py --data-dir data --image-index 0
    python scripts/demo.py --no-display --save-dir outputs/my_demo
"""

import argparse
import os
import sys
import time

# Add src to path for package imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

import torch
import numpy as np
import torchvision.transforms as T
import matplotlib
from PIL import Image

from diffvax.attack import Attack
from diffvax.immunization import DiffVaxImmunization
from diffvax.immunization.photoguard_immunization import PhotoGuardImmunization
from diffvax.immunization.diffusionguard_immunization import DiffusionGuardImmunization
from diffvax.utils import (
    set_seed_lib,
    load_image_from_path,
    save_image,
    prepare_mask_and_masked_image,
    recover_image,
)

to_pil = T.ToPILImage()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sample_image_and_mask(save_dir):
    """Create a synthetic sample image and mask for demo purposes.

    Generates a simple scene with colored rectangles and a circular mask
    so the demo can run without downloading a dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, "sample_image.png")
    mask_path = os.path.join(save_dir, "sample_mask.png")

    # Create a simple image with colored regions
    rng = np.random.RandomState(42)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(256):
        img[y, :] = [135 + y // 4, 180 + y // 8, 235]
    # Ground
    img[256:, :] = [90, 120, 80]
    # Add some rectangles as "buildings"
    img[100:256, 50:150] = [160, 140, 120]
    img[150:256, 200:280] = [140, 130, 110]
    img[80:256, 350:450] = [170, 150, 130]
    # Add noise for texture
    noise = rng.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Create a circular mask in the center
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    cy, cx = 200, 256
    Y, X = np.ogrid[:512, :512]
    circle = ((Y - cy) ** 2 + (X - cx) ** 2) <= 80 ** 2
    mask[circle] = 255

    Image.fromarray(img).save(img_path)
    Image.fromarray(mask).save(mask_path)
    return img_path, mask_path


def find_validation_image(data_dir, index):
    """Return (image_path, mask_path) for a given validation index."""
    img_dir = os.path.join(data_dir, "validation", "images")
    mask_dir = os.path.join(data_dir, "validation", "masks")
    img_path = os.path.join(img_dir, f"image_{index}.png")
    mask_path = os.path.join(mask_dir, f"mask_image_{index}.png")
    if not os.path.isfile(img_path):
        available = sorted(
            f for f in os.listdir(img_dir) if f.endswith(".png")
        ) if os.path.isdir(img_dir) else []
        raise FileNotFoundError(
            f"Validation image not found: {img_path}\n"
            f"Available images: {available}"
        )
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return img_path, mask_path


def build_comparison_figure(original, immunized, edited_orig, edited_imm, prompt, save_path=None):
    """Create a 2x2 comparison grid and optionally save it."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = [
        "Original Image",
        "Immunized Image (DiffVax)",
        f'Edited Original\n"{prompt}"',
        f'Edited Immunized\n"{prompt}"',
    ]
    images = [original, immunized, edited_orig, edited_imm]

    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("DiffVax: Immunization Against Diffusion-Based Edits", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Comparison figure saved to: {save_path}")

    return fig


def build_baseline_comparison_figure(
    original, edited_orig, edited_pg, edited_dg, edited_diffvax, prompt, save_path=None
):
    """Create a 1x5 comparison grid with all methods."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    titles = [
        "Original",
        f'Edited (No Defense)\n"{prompt}"',
        f'Edited (PhotoGuard)\n"{prompt}"',
        f'Edited (DiffusionGuard)\n"{prompt}"',
        f'Edited (DiffVax)\n"{prompt}"',
    ]
    images = [original, edited_orig, edited_pg, edited_dg, edited_diffvax]

    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle("Immunization Comparison: No Defense vs. PhotoGuard vs. DiffusionGuard vs. DiffVax",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Baseline comparison figure saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DiffVax demo: immunize an image and compare edits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/demo.py --image photo.png --mask mask.png
  python scripts/demo.py --data-dir data --image-index 2
  python scripts/demo.py --no-display --save-dir outputs/my_demo
""",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image (overrides --data-dir / --image-index)",
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help="Path to mask image (required if --image is set)",
    )
    parser.add_argument(
        "--image-index", type=int, default=0,
        help="Index of the validation image to use (default: 0)",
    )
    parser.add_argument(
        "--edit-prompt", type=str, default="a person in an alley",
        help="Text prompt for the inpainting edit (default: 'a person in an alley')",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(_project_root, "checkpoints", "diffvax_trained.pth"),
        help="Path to pre-trained DiffVax checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(_project_root, "data"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--save-dir", type=str, default=os.path.join(_project_root, "outputs", "demo"),
        help="Directory to save output images",
    )
    parser.add_argument(
        "--seed", type=int, default=5,
        help="Random seed for reproducibility (default: 5)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't show the matplotlib figure (useful for headless servers)",
    )
    parser.add_argument(
        "--attack-model", type=str, default="runwayml/stable-diffusion-inpainting",
        help="Diffusion model used for editing (default: runwayml/stable-diffusion-inpainting)",
    )
    parser.add_argument(
        "--run-baselines", action="store_true",
        help="Also run PhotoGuard and DiffusionGuard baselines for comparison",
    )
    parser.add_argument(
        "--pg-steps", type=int, default=200,
        help="PhotoGuard PGD steps (default: 200)",
    )
    parser.add_argument(
        "--dg-steps", type=int, default=100,
        help="DiffusionGuard PGD steps (default: 100)",
    )
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    print("=" * 60)
    print("  DiffVax Demo")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Resolve image and mask paths
    # ------------------------------------------------------------------
    print("[1/5] Preparing input image and mask...")
    if args.image and args.mask:
        img_path = args.image
        mask_path = args.mask
    elif args.image and not args.mask:
        parser.error("--mask is required when --image is specified")
    else:
        # Try to find validation images in data_dir
        try:
            img_path, mask_path = find_validation_image(args.data_dir, args.image_index)
        except FileNotFoundError:
            print("  Dataset not found. Generating sample image for demo...")
            sample_dir = os.path.join(args.save_dir, "sample_data")
            img_path, mask_path = create_sample_image_and_mask(sample_dir)

    print(f"  Image: {img_path}")
    print(f"  Mask:  {mask_path}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Load models
    # ------------------------------------------------------------------
    print("[2/5] Loading diffusion model and DiffVax checkpoint...")
    t0 = time.time()
    attack_model = Attack(args.attack_model)
    immunization_mdl = DiffVaxImmunization(
        attack_model,
        config={"learning_rate": 3.0},
        load_existing=True,
        load_path=args.checkpoint,
    )
    print(f"  Models loaded in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 3: Immunize the image
    # ------------------------------------------------------------------
    print("[3/5] Immunizing image...")
    t0 = time.time()
    image = load_image_from_path(img_path)
    image_mask = load_image_from_path(mask_path)
    mask_torch, image_torch, _ = prepare_mask_and_masked_image(image, image_mask)
    image_torch = image_torch.half().cuda()
    mask_torch = mask_torch.half().cuda()

    set_seed_lib(args.seed)
    immunized_img, _ = immunization_mdl.immunize_img(image_torch, mask_torch)

    adv_X = (immunized_img / 2 + 0.5).clamp(0, 1)
    adv_image_pil = to_pil(adv_X[0]).convert("RGB")
    adv_image_pil = recover_image(adv_image_pil, image, image_mask, background=True)
    print(f"  Immunization complete in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 4: Edit both original and immunized images
    # ------------------------------------------------------------------
    print(f'[4/5] Running inpainting edit: "{args.edit_prompt}"')
    t0 = time.time()

    # Edit original
    set_seed_lib(args.seed)
    edited_orig = immunization_mdl.edit_image(args.edit_prompt, image, image_mask)[0]
    edited_orig_recovered = recover_image(edited_orig, image, image_mask, background=False)

    # Edit immunized
    set_seed_lib(args.seed)
    edited_adv = immunization_mdl.edit_image(args.edit_prompt, adv_image_pil, image_mask)[0]
    edited_adv_recovered = recover_image(edited_adv, adv_image_pil, image_mask, background=False)

    print(f"  Editing complete in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 4b (optional): Run baselines
    # ------------------------------------------------------------------
    edited_pg_recovered = None
    edited_dg_recovered = None
    if args.run_baselines:
        total_steps = 6
        print(f"[5/{total_steps}] Running baselines (PhotoGuard + DiffusionGuard)...")
        t0 = time.time()

        # PhotoGuard
        pg_immunizer = PhotoGuardImmunization(
            attack_model, num_steps=args.pg_steps,
        )
        set_seed_lib(args.seed)
        pg_immunized, _ = pg_immunizer.immunize_img(image_torch, mask_torch)
        pg_X = (pg_immunized / 2 + 0.5).clamp(0, 1)
        pg_pil = to_pil(pg_X[0]).convert("RGB")
        pg_pil = recover_image(pg_pil, image, image_mask, background=True)
        print(f"  PhotoGuard immunization done ({args.pg_steps} steps)")

        set_seed_lib(args.seed)
        edited_pg = pg_immunizer.edit_image(args.edit_prompt, pg_pil, image_mask)[0]
        edited_pg_recovered = recover_image(edited_pg, pg_pil, image_mask, background=False)

        # DiffusionGuard
        dg_immunizer = DiffusionGuardImmunization(
            attack_model, num_steps=args.dg_steps, prompt=args.edit_prompt,
        )
        set_seed_lib(args.seed)
        dg_immunized, _ = dg_immunizer.immunize_img(image_torch, mask_torch)
        dg_X = (dg_immunized / 2 + 0.5).clamp(0, 1)
        dg_pil = to_pil(dg_X[0]).convert("RGB")
        dg_pil = recover_image(dg_pil, image, image_mask, background=True)
        print(f"  DiffusionGuard immunization done ({args.dg_steps} steps)")

        set_seed_lib(args.seed)
        edited_dg = dg_immunizer.edit_image(args.edit_prompt, dg_pil, image_mask)[0]
        edited_dg_recovered = recover_image(edited_dg, dg_pil, image_mask, background=False)

        print(f"  Baselines complete in {time.time() - t0:.1f}s")
        print()

    # ------------------------------------------------------------------
    # Save outputs and comparison figure
    # ------------------------------------------------------------------
    save_step = "6/6" if args.run_baselines else "5/5"
    print(f"[{save_step}] Saving results...")
    os.makedirs(args.save_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(img_path))[0]

    save_image(image, os.path.join(args.save_dir, f"{image_name}_original.png"))
    save_image(adv_image_pil, os.path.join(args.save_dir, f"{image_name}_immunized.png"))
    save_image(edited_orig_recovered, os.path.join(args.save_dir, f"{image_name}_edited_original.png"))
    save_image(edited_adv_recovered, os.path.join(args.save_dir, f"{image_name}_edited_immunized.png"))

    comparison_path = os.path.join(args.save_dir, f"{image_name}_comparison.png")
    fig = build_comparison_figure(
        image, adv_image_pil, edited_orig_recovered, edited_adv_recovered,
        args.edit_prompt, save_path=comparison_path,
    )

    figs = [fig]

    if args.run_baselines and edited_pg_recovered is not None:
        # Save baseline immunized/edited images
        save_image(pg_pil, os.path.join(args.save_dir, f"{image_name}_immunized_photoguard.png"))
        save_image(dg_pil, os.path.join(args.save_dir, f"{image_name}_immunized_diffusionguard.png"))
        save_image(edited_pg_recovered, os.path.join(args.save_dir, f"{image_name}_edited_photoguard.png"))
        save_image(edited_dg_recovered, os.path.join(args.save_dir, f"{image_name}_edited_diffusionguard.png"))

        baseline_path = os.path.join(args.save_dir, f"{image_name}_baseline_comparison.png")
        fig_bl = build_baseline_comparison_figure(
            image, edited_orig_recovered,
            edited_pg_recovered, edited_dg_recovered, edited_adv_recovered,
            args.edit_prompt, save_path=baseline_path,
        )
        figs.append(fig_bl)

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        for f in figs:
            plt.close(f)

    print()
    print("=" * 60)
    print("  Demo complete!")
    print(f"  Outputs saved to: {args.save_dir}")
    print()
    print("  Files:")
    for f in sorted(os.listdir(args.save_dir)):
        fpath = os.path.join(args.save_dir, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f} ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
