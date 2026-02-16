#!/usr/bin/env python3
"""
Generate a single multi-row comparison figure across multiple images.

Each row = one image. Columns:
  Original | Mask | No Defense | PG Encoder | PG Diffusion | DiffusionGuard | DiffVax

Usage:
    python scripts/run_comparison_figure.py
    python scripts/run_comparison_figure.py --images 1 5 9 29 33
    python scripts/run_comparison_figure.py --pg-enc-steps 200 --pg-diff-steps 50 --dg-steps 50
"""

import argparse
import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from diffvax.attack import Attack
from diffvax.immunization import DiffVaxImmunization
from diffvax.immunization.photoguard_immunization import (
    PhotoGuardImmunization,
    PhotoGuardDiffusionImmunization,
)
from diffvax.immunization.diffusionguard_immunization import DiffusionGuardImmunization
from diffvax.utils import (
    set_seed_lib,
    load_image_from_path,
    prepare_mask_and_masked_image,
    recover_image,
)

to_pil = T.ToPILImage()


def load_prompt_for_image(data_dir, image_idx):
    """Read the first prompt for a given image index from metadata.jsonl."""
    meta_path = os.path.join(data_dir, "validation", "metadata.jsonl")
    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            if row["file_name"] == f"images/image_{image_idx}.png":
                return row["prompts"][0]
    return "a person in an alley"


def immunize_and_edit(immunizer, image_torch, mask_torch, image_pil, mask_pil, prompt, seed):
    """Run immunization + edit for one method. Returns edited PIL image."""
    set_seed_lib(seed)
    imm_tensor, _ = immunizer.immunize_img(image_torch, mask_torch)
    imm_pil = to_pil((imm_tensor / 2 + 0.5).clamp(0, 1)[0]).convert("RGB")
    imm_pil = recover_image(imm_pil, image_pil, mask_pil, background=True)
    set_seed_lib(seed)
    edited = immunizer.edit_image(prompt, imm_pil, mask_pil)[0]
    edited_recovered = recover_image(edited, imm_pil, mask_pil, background=False)
    torch.cuda.empty_cache()
    return edited_recovered


def run_one_image(
    image_idx, data_dir, attack_model, diffvax_model,
    pg_enc, pg_diff, dg, prompt, seed,
):
    """Run all methods on one image. Returns dict of PIL images."""
    img_path = os.path.join(data_dir, "validation", "images", f"image_{image_idx}.png")
    mask_path = os.path.join(data_dir, "validation", "masks", f"mask_image_{image_idx}.png")

    image_pil = load_image_from_path(img_path)
    mask_pil = load_image_from_path(mask_path)
    mask_torch, _, image_torch = prepare_mask_and_masked_image(image_pil, mask_pil)
    image_torch = image_torch.half().cuda()
    mask_torch = mask_torch.half().cuda()

    results = {"original": image_pil, "mask": mask_pil}

    # No defense edit
    set_seed_lib(seed)
    edited_orig = diffvax_model.edit_image(prompt, image_pil, mask_pil)[0]
    results["no_defense"] = recover_image(edited_orig, image_pil, mask_pil, background=False)

    # PhotoGuard Encoder
    print("    PhotoGuard Encoder...")
    results["pg_encoder"] = immunize_and_edit(
        pg_enc, image_torch, mask_torch, image_pil, mask_pil, prompt, seed,
    )

    # PhotoGuard Diffusion
    print("    PhotoGuard Diffusion...")
    pg_diff.prompt = prompt
    results["pg_diffusion"] = immunize_and_edit(
        pg_diff, image_torch, mask_torch, image_pil, mask_pil, prompt, seed,
    )

    # DiffusionGuard
    print("    DiffusionGuard...")
    dg.prompt = prompt
    results["diffusionguard"] = immunize_and_edit(
        dg, image_torch, mask_torch, image_pil, mask_pil, prompt, seed,
    )

    # DiffVax
    print("    DiffVax...")
    results["diffvax"] = immunize_and_edit(
        diffvax_model, image_torch, mask_torch, image_pil, mask_pil, prompt, seed,
    )

    return results


def build_grid_figure(all_results, prompts, save_path):
    """Build N-row x 6-col comparison figure."""
    n_rows = len(all_results)
    col_keys = [
        "original", "mask", "no_defense",
        "pg_encoder", "pg_diffusion",
        "diffusionguard", "diffvax",
    ]
    col_labels = [
        "Original", "Mask", "Edited\n(No Defense)",
        "Edited\n(PG Encoder)", "Edited\n(PG Diffusion)",
        "Edited\n(DiffusionGuard)", "Edited\n(DiffVax)",
    ]
    n_cols = len(col_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row_idx, (results, prompt) in enumerate(zip(all_results, prompts)):
        for col_idx, (key, label) in enumerate(zip(col_keys, col_labels)):
            ax = axes[row_idx, col_idx]
            ax.imshow(results[key])
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        # Prompt as row label
        axes[row_idx, 0].text(
            -0.05, 0.5, f'"{prompt}"',
            transform=axes[row_idx, 0].transAxes,
            fontsize=9, fontstyle="italic", va="center", ha="right",
            rotation=90,
        )

    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.savefig(save_path, bbox_inches="tight", dpi=200, pad_inches=0.1)
    plt.close(fig)
    print(f"\nFigure saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-image baseline comparison figure")
    parser.add_argument("--images", type=int, nargs="+", default=[1, 5, 9, 29, 33],
                        help="Validation image indices to use")
    parser.add_argument("--data-dir", type=str, default=os.path.join(_project_root, "data"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(_project_root, "checkpoints", "diffvax_trained.pth"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(_project_root, "outputs", "baseline_comparison.png"))
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--pg-enc-steps", type=int, default=200,
                        help="PhotoGuard encoder attack PGD steps (default: 200)")
    parser.add_argument("--pg-diff-steps", type=int, default=50,
                        help="PhotoGuard diffusion attack PGD steps (default: 50)")
    parser.add_argument("--pg-diff-grad-reps", type=int, default=10,
                        help="PhotoGuard diffusion gradient averaging reps (default: 10)")
    parser.add_argument("--dg-steps", type=int, default=50,
                        help="DiffusionGuard PGD steps (default: 50)")
    parser.add_argument("--attack-model", type=str, default="runwayml/stable-diffusion-inpainting")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Image Baseline Comparison")
    print(f"  Images: {args.images}")
    print(f"  PG Encoder steps: {args.pg_enc_steps}")
    print(f"  PG Diffusion steps: {args.pg_diff_steps} (grad_reps: {args.pg_diff_grad_reps})")
    print(f"  DG steps: {args.dg_steps}")
    print("=" * 60)
    print()

    # Load models once
    print("Loading models...")
    t0 = time.time()
    attack_model = Attack(args.attack_model)
    diffvax_model = DiffVaxImmunization(
        attack_model, config={"learning_rate": 3.0},
        load_existing=True, load_path=args.checkpoint,
    )
    pg_enc = PhotoGuardImmunization(attack_model, num_steps=args.pg_enc_steps)
    pg_diff = PhotoGuardDiffusionImmunization(
        attack_model, num_steps=args.pg_diff_steps,
        grad_reps=args.pg_diff_grad_reps,
    )
    dg = DiffusionGuardImmunization(attack_model, num_steps=args.dg_steps)
    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    # Process each image
    all_results = []
    prompts = []
    for i, idx in enumerate(args.images):
        prompt = load_prompt_for_image(args.data_dir, idx)
        prompts.append(prompt)
        print(f"[{i+1}/{len(args.images)}] image_{idx} — \"{prompt}\"")
        t0 = time.time()
        results = run_one_image(
            idx, args.data_dir, attack_model, diffvax_model,
            pg_enc, pg_diff, dg, prompt, args.seed,
        )
        all_results.append(results)
        print(f"  Done in {time.time() - t0:.1f}s\n")

    # Build figure
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_grid_figure(all_results, prompts, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
