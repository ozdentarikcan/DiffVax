"""DiffusionGuard baseline immunization.

Implements the early-stage noise maximization approach from:
  Li et al., "DiffusionGuard: A Robust Defense Against Malicious Diffusion-based
  Image Editing", ICLR 2025.

PGD optimization that maximizes ||ε_θ(x_T; text, T, mask, x+δ)||₂² — the UNet's
noise prediction at the highest timestep. Uses mask augmentation for robustness.
"""

import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import gc


class DiffusionGuardImmunization:
    def __init__(
        self,
        attack_model,
        eps=16 / 255,
        step_size=1 / 255,
        num_steps=800,
        num_mask_augmentations=4,
        prompt="",
    ):
        self.attack_model = attack_model
        # Papers use eps in [0,255] space; codebase uses [-1,1] so scale by 2
        self.eps = eps * 2
        self.step_size = step_size * 2
        self.num_steps = num_steps
        self.num_mask_augmentations = num_mask_augmentations
        self.prompt = prompt
        self.clamp_min = -1
        self.clamp_max = 1

    def _augment_mask(self, mask, contour_eps=10, contour_sigma=3.0,
                      contour_iters=15):
        """Random contour perturbation for mask augmentation.

        Matches the contour-based augmentation from the original DiffusionGuard
        implementation (Li et al., ICLR 2025). Finds contours in the mask,
        applies random Gaussian-smoothed offsets to contour points, and
        constrains perturbed points to lie within the original mask.

        Args:
            mask: tensor of shape (1, 1, H, W) with values in {0, 1}.
                  Convention: 1 = edit region.
            contour_eps: max random offset applied to contour points.
            contour_sigma: std-dev for Gaussian smoothing of offsets.
            contour_iters: max iterations of contour perturbation
                (actual count is randomly sampled from [1, contour_iters]).

        Returns:
            Augmented mask tensor with same shape and dtype.
        """
        mask_2d = mask[0, 0].cpu().numpy().astype(np.uint8)
        # The reference works on the inverted mask (1 = face/keep region)
        inverted = 1 - mask_2d

        iters = int(random.uniform(1, contour_iters))
        for _ in range(iters):
            contours, _ = cv2.findContours(
                inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                break

            contour = max(contours, key=cv2.contourArea)
            contour_np = contour.reshape(-1, 2)
            num_points = len(contour_np)

            if contour_eps < 1:
                break

            # Random offsets smoothed by a Gaussian filter
            offsets_x = np.random.randint(
                -contour_eps, contour_eps + 1, size=num_points
            )
            offsets_y = np.random.randint(
                -contour_eps, contour_eps + 1, size=num_points
            )
            offsets_x = gaussian_filter(offsets_x.astype(np.float64),
                                        sigma=contour_sigma)
            offsets_y = gaussian_filter(offsets_y.astype(np.float64),
                                        sigma=contour_sigma)

            perturbed = contour_np.copy()
            perturbed[:, 0] += offsets_x.astype(int)
            perturbed[:, 1] += offsets_y.astype(int)

            # Clip to image bounds
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0,
                                      mask_2d.shape[1] - 1)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0,
                                      mask_2d.shape[0] - 1)

            # Snap points that escaped the original mask back to the nearest
            # original contour point
            for i in range(num_points):
                x, y = perturbed[i]
                if inverted[int(y), int(x)] == 0:
                    dists = np.sqrt(
                        (contour_np[:, 0] - x) ** 2
                        + (contour_np[:, 1] - y) ** 2
                    )
                    perturbed[i] = contour_np[np.argmin(dists)]

            modified = np.zeros_like(inverted)
            cv2.drawContours(
                modified,
                [perturbed.reshape(-1, 1, 2).astype(int)],
                0, 1, cv2.FILLED,
            )
            inverted = modified

        augmented = (1 - inverted).astype(np.float32)
        return torch.from_numpy(augmented)[None, None].to(
            device=mask.device, dtype=mask.dtype
        )

    def immunize_img(self, img, img_mask):
        """Apply DiffusionGuard noise-maximization attack to protect image.

        Args:
            img: image tensor in [-1,1], shape (1, 3, H, W), half precision on CUDA.
            img_mask: mask tensor, shape (1, 1, H, W). 1 = edit region.

        Returns:
            (immunized_tensor, perturbation) both in [-1,1] half precision.
        """
        pipe = self.attack_model.model
        vae = pipe.vae
        unet = pipe.unet
        scheduler = pipe.scheduler

        img_f = img.float()
        mask_f = img_mask.float()

        # Get text embeddings (unconditional + conditional)
        text_embeddings = self.attack_model.tokenize_prompt(
            pipe, self.prompt, batch_size=1
        )

        # Get max timestep (num_train_timesteps - 1, since alphas_cumprod is 0-indexed)
        max_timestep = torch.tensor(
            scheduler.config.num_train_timesteps - 1,
            device=pipe.device,
            dtype=torch.long,
        )

        # PGD optimization
        delta = torch.zeros_like(img_f, requires_grad=True)

        for step in range(self.num_steps):
            total_grad = torch.zeros_like(img_f)

            for aug_idx in range(self.num_mask_augmentations):
                # Augment mask
                if aug_idx == 0:
                    aug_mask = mask_f
                else:
                    aug_mask = self._augment_mask(mask_f)

                perturbed = torch.clamp(
                    img_f + delta, self.clamp_min, self.clamp_max
                )

                # Encode perturbed image through VAE
                perturbed_latents = vae.encode(
                    perturbed.half()
                ).latent_dist.sample().float()
                perturbed_latents = 0.18215 * perturbed_latents

                # Prepare masked image latents
                masked_image = perturbed * (aug_mask < 0.5).float()
                masked_image_latents = vae.encode(
                    masked_image.half()
                ).latent_dist.sample().float()
                masked_image_latents = 0.18215 * masked_image_latents

                # Downsample mask to latent resolution
                mask_latent = F.interpolate(
                    aug_mask, size=(img_f.shape[2] // 8, img_f.shape[3] // 8)
                )

                # Add noise at max timestep
                noise = torch.randn_like(perturbed_latents)
                noisy_latents = scheduler.add_noise(
                    perturbed_latents, noise, max_timestep
                )

                # Build 9-channel UNet input: noisy_latents + mask + masked_image_latents
                latent_model_input = torch.cat(
                    [noisy_latents, mask_latent, masked_image_latents], dim=1
                )

                # UNet forward — use only conditional embeddings (no CFG for efficiency)
                cond_embeddings = text_embeddings[1:2]  # conditional only
                noise_pred = unet(
                    latent_model_input.half(),
                    max_timestep,
                    encoder_hidden_states=cond_embeddings,
                ).sample.float()

                # Loss: maximize noise prediction norm (minimize negative)
                loss = -(noise_pred ** 2).sum()
                loss.backward()

                with torch.no_grad():
                    total_grad += delta.grad.clone()

                delta.grad.zero_()

                del (
                    perturbed_latents, masked_image_latents, mask_latent,
                    noise, noisy_latents, latent_model_input, noise_pred, loss,
                )

            # Sign-gradient descent with accumulated gradients
            with torch.no_grad():
                grad_sign = total_grad.sign()
                delta.data = delta.data - self.step_size * grad_sign
                # Project to L∞ ball
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                # Apply perturbation only outside mask (1 - mask)
                delta.data = delta.data * (1 - mask_f)
                # Ensure perturbed image stays in valid range
                delta.data = torch.clamp(
                    img_f + delta.data, self.clamp_min, self.clamp_max
                ) - img_f

            if step % 25 == 0:
                torch.cuda.empty_cache()

        immunized = torch.clamp(img_f + delta.detach(), self.clamp_min, self.clamp_max)
        perturbation = (immunized - img_f) * (1 - mask_f)

        del delta, total_grad, text_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        return immunized.half(), perturbation.half()

    def edit_image(self, prompt, img, img_mask, num_inf=30, SEED=5):
        """Edit image using the diffusion inpainting pipeline."""
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        edited_image = self.attack_model.model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=7.5,
            strength=1.0,
            generator=generator,
        ).images
        return edited_image
