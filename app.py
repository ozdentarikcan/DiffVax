#!/usr/bin/env python3
"""Gradio demo for DiffVax immunization."""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "src"))

import torch
import torchvision.transforms as T
import numpy as np
import gradio as gr
from PIL import Image

from diffvax.attack import Attack
from diffvax.immunization import DiffVaxImmunization
from diffvax.utils import (
    set_seed_lib,
    prepare_mask_and_masked_image,
    recover_image,
)

PROJECT_ROOT = _script_dir
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "diffvax_trained.pth")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_ID = "runwayml/stable-diffusion-inpainting"

to_pil = T.ToPILImage()

# ---------------------------------------------------------------------------
# Global model state (loaded once)
# ---------------------------------------------------------------------------
attack_model = None
diffvax_model = None


def load_models():
    global attack_model, diffvax_model
    if attack_model is not None:
        return
    print("Loading models...")
    attack_model = Attack(MODEL_ID)
    diffvax_model = DiffVaxImmunization(
        attack_model,
        config={"learning_rate": 3.0},
        load_existing=True,
        load_path=CHECKPOINT,
    )
    print("Models loaded.")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_diffvax(editor_value, mask_upload, prompt, seed):
    """Run DiffVax immunization + editing. Returns 3 images."""
    load_models()
    seed = int(seed)

    # ImageEditor returns dict with "background", "layers", and "composite"
    if editor_value is None or editor_value.get("background") is None:
        raise gr.Error("Please upload an image first.")

    image_pil = editor_value["background"].convert("RGB")

    # Mask priority: uploaded mask > painted brush strokes
    # Model expects: white (255) = edit region, black (0) = protected region
    if mask_upload is not None:
        mask_pil = mask_upload.convert("RGB")
    else:
        # User paints the PROTECTED area (black brush).
        # Alpha channel tells us where user painted → those areas are protected.
        # Invert so: painted (protected) = black, unpainted (edit) = white.
        layers = editor_value.get("layers", [])
        if layers and len(layers) > 0:
            painted_np = np.zeros((image_pil.height, image_pil.width), dtype=np.float32)
            for layer in layers:
                alpha = np.array(layer.getchannel("A"), dtype=np.float32) / 255.0
                painted_np = np.clip(painted_np + alpha, 0, 1)
            # Invert: painted = 0 (protected), unpainted = 1 (edit region)
            mask_np = 1.0 - painted_np
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("RGB")
        else:
            raise gr.Error("Please paint the protected area or upload a mask image.")

    # Resize to 512x512 (model requirement)
    image_pil = image_pil.resize((512, 512))
    mask_pil = mask_pil.resize((512, 512))

    # Prepare tensors
    mask_torch, image_torch, _ = prepare_mask_and_masked_image(image_pil, mask_pil)
    image_torch = image_torch.half().cuda()
    mask_torch = mask_torch.half().cuda()

    # Immunize
    set_seed_lib(seed)
    immunized_tensor, _ = diffvax_model.immunize_img(image_torch, mask_torch)
    imm_pil = to_pil((immunized_tensor / 2 + 0.5).clamp(0, 1)[0]).convert("RGB")
    imm_pil = recover_image(imm_pil, image_pil, mask_pil, background=True)

    # Edit original
    set_seed_lib(seed)
    edited_orig = diffvax_model.edit_image(prompt, image_pil, mask_pil)[0]
    edited_orig = recover_image(edited_orig, image_pil, mask_pil, background=False)

    # Edit immunized
    set_seed_lib(seed)
    edited_imm = diffvax_model.edit_image(prompt, imm_pil, mask_pil)[0]
    edited_imm = recover_image(edited_imm, imm_pil, mask_pil, background=False)

    torch.cuda.empty_cache()

    return imm_pil, edited_orig, edited_imm


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(
        title="DiffVax Demo",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# DiffVax: Immunization Against Diffusion-Based Image Editing\n"
            "Upload an image and mask, enter an editing prompt, and see how "
            "DiffVax protects the image from unauthorized inpainting edits."
        )

        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    label="Input Image",
                    type="pil",
                    height=450,
                    brush=gr.Brush(
                        colors=["#000000"],
                        default_color="#000000",
                        default_size=30,
                    ),
                    eraser=gr.Eraser(default_size=30),
                    sources=["upload"],
                    transforms=[],
                    layers=False,
                )
                gr.Markdown(
                    "*Upload an image, then paint the area to protect "
                    "with the brush (black = protected).*"
                )
                mask_upload = gr.Image(
                    label="Or upload a mask (white = edit region, black = protected; overrides brush)",
                    type="pil",
                    height=200,
                )
                prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="e.g. A person in a weekend market",
                    lines=1,
                )
                seed = gr.Number(label="Seed", value=5, precision=0)
                run_btn = gr.Button("Run DiffVax", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                with gr.Row():
                    out_immunized = gr.Image(label="Immunized Image", height=350)
                with gr.Row():
                    out_edited_orig = gr.Image(label="Edited Original (No Defense)", height=350)
                    out_edited_imm = gr.Image(label="Edited Immunized (DiffVax)", height=350)

        run_btn.click(
            fn=run_diffvax,
            inputs=[editor, mask_upload, prompt, seed],
            outputs=[out_immunized, out_edited_orig, out_edited_imm],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
