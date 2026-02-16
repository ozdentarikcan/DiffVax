"""DiffVax immunization against diffusion attacks."""

import torch
import numpy as np
import torchvision.transforms as T
import os
import datetime
from tqdm import tqdm
from PIL import Image
import gc

from diffvax.model import NestedUNet
from diffvax.utils import set_seed_lib

scaler = torch.cuda.amp.GradScaler()


class ImmunizationDataset(torch.utils.data.Dataset):
    """Dataset for immunization training."""

    def __init__(self, img_list, img_mask_list, prompt_list):
        self.img_list = img_list
        self.img_mask_list = img_mask_list
        self.prompt_list = prompt_list

    def __getitem__(self, index):
        img = self.img_list[index]
        img_mask = self.img_mask_list[index]
        prompt = self.prompt_list[index]
        return img, img_mask, prompt

    def __len__(self):
        return len(self.img_list)


class DiffVaxImmunization:
    def __init__(
        self,
        attack_model,
        config,
        load_existing=False,
        existing_iter_num=0,
        load_path=None,
        output_dir=None,
    ):
        self.model_name = "DiffVaxImmunization"
        self.attack_model = attack_model
        self.step_size = 1
        self.eps = 32 / 255
        self.clamp_min = -1
        self.clamp_max = 1
        self.output_dir = output_dir or "outputs"

        unetmodel = NestedUNet(num_classes=3)
        self.unetmodel = unetmodel.to("cuda")
        learning_rate = config["learning_rate"]
        self.optimizer = torch.optim.Adam(unetmodel.parameters(), lr=learning_rate)

        self.load_existing = load_existing
        self.existing_iter_num = existing_iter_num

        if self.load_existing:
            self.unetmodel.load_state_dict(torch.load(load_path, weights_only=True))
        self.model = self.unetmodel

        for param in self.unetmodel.parameters():
            param.requires_grad = True

        generator = torch.Generator(device="cuda")
        self.generator = generator

    def immunize_img(self, img, img_mask, epsilon=32):
        """Apply immunization perturbation to image."""
        img_f = img.float().cuda()
        unet_out = self.unetmodel.forward(img_f)
        unet_out = unet_out.half().cuda() * (1 - img_mask)

        img_adv = torch.clamp(img + unet_out, self.clamp_min, self.clamp_max)

        return img_adv, unet_out

    def train_immunization_all_images_batch(
        self,
        img_list,
        img_mask_list,
        prompt_list,
        target_image=None,
        iter_num=2000,
        SEED=5,
        batch_size=2,
        alpha=1,
        loss_type="l2",
    ):
        set_seed_lib(SEED)
        total_iter = 0

        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        existing_folders = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()
        ]
        last_idx = max([int(x) for x in existing_folders], default=0) + 1
        run_dir = os.path.join(models_dir, str(last_idx))
        os.makedirs(run_dir, exist_ok=True)

        if self.load_existing:
            path_of_models = os.path.join(
                models_dir,
                f"sd15_all_images_half_mult_img_mult_prompt_immunization_model_{self.model_name}_iter_{iter_num + self.existing_iter_num}_alpha_{alpha}_loss_{loss_type}",
            )
        else:
            path_of_models = os.path.join(
                run_dir,
                f"sd15_all_images_{self.model_name}_iter_{iter_num}_alpha_{alpha}_loss_{loss_type}_batch_{batch_size}",
            )

        dataset = ImmunizationDataset(img_list, img_mask_list, prompt_list)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch_i in range(iter_num):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            epoch_losses = []
            epoch_losses1 = []
            epoch_losses2 = []

            for i, (img_batch, mask_batch, prompt_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                losses = []
                losses1 = []
                losses2 = []
                cur_iter = i + self.existing_iter_num

                mask_batch.requires_grad = False
                img_batch.requires_grad_()

                img_f = img_batch.float().cuda()
                unet_out = self.unetmodel.forward(img_f)

                unet_out = unet_out.half().cuda() * (1 - mask_batch)
                img_adv = torch.clamp(
                    img_batch + unet_out, self.clamp_min, self.clamp_max
                )
                img_out = self.attack_model.attack(
                    prompt=prompt_batch,
                    masked_image=img_adv,
                    mask=mask_batch,
                    num_inference_steps=4,
                    batch_size=batch_size,
                )

                target_image = torch.zeros_like(img_out).cuda()

                loss1 = (((img_out - target_image) * (mask_batch / 512)).norm(p=1) / (mask_batch / 512).sum())
                loss2 = (alpha * (img_adv - img_batch) * ((1 - mask_batch) / 512)).norm(p=1) / ((1 - mask_batch) / 512).sum()
                loss = loss1 + loss2

                loss1 = loss1.item()
                loss2 = loss2.item()

                losses.append(loss.item())
                losses1.append(loss1)
                losses2.append(loss2)
                epoch_losses.append(loss.item())
                epoch_losses1.append(loss1)
                epoch_losses2.append(loss2)

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                torch.cuda.empty_cache()
                gc.collect()

                total_iter += batch_size

                pbar.set_description_str(
                    f"AVG Loss: {np.mean(losses):.3f} Loss1: {np.mean(losses1):.3f} Loss2: {np.mean(losses2):.3f}"
                )
                pbar.update(1)
                if torch.isnan(loss):
                    torch.save(
                        self.model.state_dict(),
                        path_of_models + f"iter_{cur_iter}_early.pth",
                    )
                    return
                losses = []
                losses1 = []
                losses2 = []

        torch.save(self.model.state_dict(), path_of_models + "_final.pth")

        return img_adv, path_of_models + "_final.pth"

    def edit_image(
        self,
        prompt,
        img,
        img_mask,
        num_inf=30,
        SEED=5,
        generator=None,
    ):
        """Edit image using the diffusion model."""
        strength = 1.0
        guidance_scale = 7.5
        self.generator.manual_seed(SEED)

        edited_image = self.attack_model.model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=self.generator,
        ).images

        return edited_image
