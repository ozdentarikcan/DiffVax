"""Calculate image quality metrics for immunization evaluation."""

import numpy as np
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.metrics import MetricType, create_metric
from diffvax.utils import set_seed_lib, recover_image

SEED = 5


def take_background(image, mask):
    """Extract the masked (edit) region from image."""
    return recover_image(image, image, mask, background=False)


def calculate_metrics_for_image(
    orig_image, adv_image, image_mask, prompt_list, image_name,
    immunization_model, save_path, no_metric=False, sampling_steps=30,
):
    noise_metrics = {}
    edit_metrics = {}
    noise_metrics["psnr"] = create_metric(MetricType.PSNR)
    noise_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["psnr"] = create_metric(MetricType.PSNR)
    edit_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["fsim"] = create_metric(MetricType.FSIM)
    edit_metrics["clip"] = create_metric(MetricType.CLIP, model='ViT-B-32', pretrained_on='laion2b_s34b_b79k')

    log = {}
    log["image_name"] = image_name
    log["prompt_list"] = prompt_list
    log["immunization_model"] = immunization_model.model_name
    orig_image_np = np.array(orig_image.convert("RGB"))
    print(image_name)
    adv_image_np = np.array(adv_image.convert("RGB"))
    noise_metric_values = {}
    # Difference between the output (image+noise) and the original image: PSNR and SSIM
    psnr_noise = noise_metrics["psnr"]([orig_image_np], [adv_image_np])[0]
    ssim_noise = noise_metrics["ssim"]([orig_image_np], [adv_image_np])[0]
    noise_metric_values["psnr"] = psnr_noise
    noise_metric_values["ssim"] = ssim_noise

    edit_metric_values = {}
    for metric_name in edit_metrics.keys():
        edit_metric_values[metric_name] = {}

    # Difference between the edited output (image+noise) and the edited original image
    for prompt_ind, prompt in enumerate(prompt_list):
        set_seed_lib(SEED)
        edited_adv = immunization_model.edit_image(prompt, adv_image, image_mask, num_inf=sampling_steps)[0]
        set_seed_lib(SEED)
        edited_orig = immunization_model.edit_image(prompt, orig_image, image_mask, num_inf=sampling_steps)[0]

        edited_adv_background = take_background(edited_adv, image_mask)
        edited_orig_background = take_background(edited_orig, image_mask)
        edited_adv_recovered = recover_image(edited_adv, adv_image, image_mask, background=False)
        edited_orig_recovered = recover_image(edited_orig, orig_image, image_mask, background=False)
        adv_dir = os.path.join(save_path, 'images')
        os.makedirs(adv_dir, exist_ok=True)
        edited_adv_recovered.save(os.path.join(adv_dir, f"{image_name}_prompt_{prompt_ind}_edited_result_adv.png"))
        edited_orig_recovered.save(os.path.join(adv_dir, f"{image_name}_prompt_{prompt_ind}_edited_result_orig.png"))

        if no_metric:
            continue
        for metric_name, edit_metric in edit_metrics.items():
            if metric_name == "clip":
                adv_metric = edit_metric([edited_adv_background], [prompt])[0]
            else:
                edited_adv_background_np = np.array(edited_adv_background.convert("RGB"))
                edited_orig_background_np = np.array(edited_orig_background.convert("RGB"))
                adv_metric = edit_metric([edited_orig_background_np], [edited_adv_background_np])[0]
            edit_metric_values[metric_name][prompt] = adv_metric

    return noise_metric_values, edit_metric_values
