#!/usr/bin/env python3
"""Train DiffVax immunization model."""

import argparse
import os
import sys
import yaml

# Add src to path for package imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.attack import Attack
from diffvax.immunization import DiffVaxImmunization
from diffvax.utils import (
    load_image,
    prepare_mask_and_masked_image,
    get_train_val_image_prompt_list,
    ensure_dataset_in_data_dir,
)


def immunize_image_list(image_prompt_list, config, data_dir, output_dir):
    iter_num = config["iter_num"]
    immunization_model_name = config["immunization_model"]
    alpha = config["alpha"]
    batch_size = config["batch_size"]
    train_all = config["train_all"]
    attack_model_link = config["attack_model_link"]

    attack_model = Attack(attack_model_link)

    immunization_config = {
        "iter_num": iter_num,
        "learning_rate": config["learning_rate"],
        "immunization_model": immunization_model_name,
    }
    immunization_mdl = DiffVaxImmunization(
        attack_model, immunization_config, output_dir=output_dir
    )

    if not train_all:
        index_list = config["image_index_list"]
        image_prompt_list = [image_prompt_list[index] for index in index_list]

    image_name_list = [image_prompt["image"][:-4] for image_prompt in image_prompt_list]
    prompt_list = [image_prompt["prompts"] for image_prompt in image_prompt_list]
    image_torch_list = []
    mask_torch_list = []
    prompt_train_list = []

    # Support both dataset layouts: images/masks or cropped_images/sam_masks
    images_subdir = config.get("images_subdir", "train/images")
    masks_subdir = config.get("masks_subdir", "train/masks")

    for image_ind, image_name in enumerate(image_name_list):
        image = load_image(
            image_name,
            data_dir,
            is_mask=False,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
        )
        image_mask = load_image(
            image_name,
            data_dir,
            is_mask=True,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
        )
        mask_torch, image_torch, non_masked_image_torch = prepare_mask_and_masked_image(
            image, image_mask
        )
        image_torch = image_torch.half().cuda()
        non_masked_image_torch = non_masked_image_torch.half().cuda()
        mask_torch = mask_torch.half().cuda()

        cur_prompt_list = prompt_list[image_ind]
        for prompt in cur_prompt_list:
            image_torch_list.append(image_torch.squeeze(0))
            mask_torch_list.append(mask_torch.squeeze(0))
            prompt_train_list.append(prompt)

    immunized_img, immunization_model_path = (
        immunization_mdl.train_immunization_all_images_batch(
            image_torch_list,
            mask_torch_list,
            prompt_train_list,
            target_image=None,
            alpha=alpha,
            iter_num=iter_num,
            batch_size=batch_size,
        )
    )

    return immunization_model_path


def main():
    parser = argparse.ArgumentParser(description="Train DiffVax immunization model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir

    data_dir = config["data_dir"]

    data_dir = ensure_dataset_in_data_dir(
        repo_id="ozdentarikcan/DiffVaxDataset",
        data_dir=data_dir,
    )

    train_list, val_list = get_train_val_image_prompt_list(data_dir)

    immunization_model_path = immunize_image_list(
        train_list, config, data_dir, config["output_dir"]
    )
    print(f"Training complete. Model saved to: {immunization_model_path}")


if __name__ == "__main__":
    main()
