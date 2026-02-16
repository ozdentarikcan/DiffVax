"""PhotoGuard baseline immunization (Encoder + Diffusion attacks).

Implements both attack variants from:
  Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing",
  ICML 2023 (MadryLab).

Encoder Attack (simple):
  PGD with L∞ constraint targeting the VAE encoder. Forces
  VAE.encode(x+δ).latent_dist.mean toward a target (zeros).
  Uses decaying step size matching the original implementation.

Diffusion Attack (complex):
  Full differentiable forward pass through the inpainting pipeline.
  L2-norm PGD with gradient averaging over multiple stochastic forward
  passes to handle diffusion randomness.
"""

import torch
import numpy as np
import gc
from tqdm import tqdm


class PhotoGuardImmunization:
    """PhotoGuard Encoder Attack (simple attack).

    PGD against VAE.encode().latent_dist.mean with L∞ constraint and
    decaying step size, matching the original MadryLab implementation.
    """

    def __init__(self, attack_model, eps=0.06, step_size=0.01, num_steps=1000):
        self.attack_model = attack_model
        self.eps = eps          # already in [-1,1] space
        self.step_size = step_size
        self.num_steps = num_steps
        self.clamp_min = -1
        self.clamp_max = 1

    def immunize_img(self, img, img_mask):
        """Apply PhotoGuard encoder attack.

        Args:
            img: image tensor in [-1,1], shape (1, 3, H, W), half on CUDA.
            img_mask: mask tensor, shape (1, 1, H, W). 1 = edit region.

        Returns:
            (immunized_tensor, perturbation) both in [-1,1] half precision.
        """
        vae = self.attack_model.model.vae
        img_f = img.float()
        mask_f = img_mask.float()

        # Target: encode zeros (gray/black), use .mean for determinism
        with torch.no_grad():
            target = vae.encode(
                torch.zeros_like(img_f).half()
            ).latent_dist.mean.float()

        # Initialize with random perturbation inside eps ball
        X_adv = img_f.clone().detach() + (
            torch.rand_like(img_f) * 2 * self.eps - self.eps
        )
        X_adv = torch.clamp(X_adv, self.clamp_min, self.clamp_max)

        pbar = tqdm(range(self.num_steps), desc="[PhotoGuard Encoder]")
        for i in pbar:
            # Decaying step size matching original
            actual_step_size = self.step_size - (
                self.step_size - self.step_size / 100
            ) / self.num_steps * i

            X_adv.requires_grad_(True)
            loss = (
                vae.encode(X_adv.half()).latent_dist.mean.float() - target
            ).norm()

            pbar.set_description(
                f"[PhotoGuard Encoder] Loss {loss.item():.4f} | "
                f"step {actual_step_size:.4f}"
            )

            grad = torch.autograd.grad(loss, [X_adv])[0]

            X_adv = X_adv - grad.detach().sign() * actual_step_size
            X_adv = torch.minimum(
                torch.maximum(X_adv, img_f - self.eps), img_f + self.eps
            )
            X_adv.data = torch.clamp(X_adv, self.clamp_min, self.clamp_max)

            # Apply perturbation only outside mask (1 - mask)
            X_adv.data = X_adv.data * (1 - mask_f) + img_f * mask_f

            X_adv.grad = None

            if i % 100 == 0:
                torch.cuda.empty_cache()

        immunized = X_adv.detach()
        perturbation = (immunized - img_f) * (1 - mask_f)

        del target
        torch.cuda.empty_cache()
        gc.collect()

        return immunized.half(), perturbation.half()

    def edit_image(self, prompt, img, img_mask, num_inf=30, SEED=5):
        """Edit image using the diffusion inpainting pipeline."""
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return self.attack_model.model(
            prompt=prompt, image=img, mask_image=img_mask,
            eta=1, num_inference_steps=num_inf,
            guidance_scale=7.5, strength=1.0, generator=generator,
        ).images


class PhotoGuardDiffusionImmunization:
    """PhotoGuard Diffusion Attack (complex attack).

    Full differentiable forward pass through the inpainting pipeline with
    L2-norm PGD and gradient averaging over multiple stochastic passes.
    Matches the original MadryLab implementation.
    """

    def __init__(
        self,
        attack_model,
        eps=16.0,
        step_size=1.0,
        num_steps=200,
        grad_reps=10,
        attack_inference_steps=4,
        prompt="",
    ):
        self.attack_model = attack_model
        self.eps = eps            # L2 budget (in [-1,1] pixel space)
        self.step_size = step_size
        self.num_steps = num_steps
        self.grad_reps = grad_reps
        self.attack_inference_steps = attack_inference_steps
        self.prompt = prompt
        self.clamp_min = -1
        self.clamp_max = 1

    def _attack_forward(self, masked_image, mask, prompt, guidance_scale=7.5, eta=1):
        """Differentiable forward pass through inpainting pipeline.

        Mirrors attack_forward() from the original PhotoGuard codebase.
        """
        pipe = self.attack_model.model

        # Tokenize
        text_embeddings = self.attack_model.tokenize_prompt(
            pipe, prompt, batch_size=1
        )

        num_channels_latents = pipe.vae.config.latent_channels
        latents_shape = (1, num_channels_latents, 64, 64)
        latents = torch.randn(
            latents_shape, device=pipe.device, dtype=text_embeddings.dtype
        )

        mask_down = torch.nn.functional.interpolate(mask, size=(64, 64))
        mask_down = torch.cat([mask_down] * 2)

        masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        latents = latents * pipe.scheduler.init_noise_sigma

        pipe.scheduler.set_timesteps(self.attack_inference_steps)
        timesteps = pipe.scheduler.timesteps.to(pipe.device)

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat(
                [latent_model_input, mask_down, masked_image_latents], dim=1
            )
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = pipe.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        return image

    def _compute_grad(self, cur_mask, cur_masked_image, target_image):
        """One stochastic forward pass + backprop."""
        torch.set_grad_enabled(True)
        cur_masked_image = cur_masked_image.clone().requires_grad_(True)

        with torch.autocast("cuda"):
            image_out = self._attack_forward(
                cur_masked_image, cur_mask, self.prompt,
            )

        loss = (image_out.float() - target_image).norm(p=2)
        grad = torch.autograd.grad(loss, [cur_masked_image])[0]
        # Only perturb outside mask
        grad = grad * (1 - cur_mask.float())

        return grad, loss.item()

    def immunize_img(self, img, img_mask):
        """Apply PhotoGuard diffusion attack (L2 PGD).

        Args:
            img: image tensor in [-1,1], shape (1, 3, H, W), half on CUDA.
            img_mask: mask tensor, shape (1, 1, H, W). 1 = edit region.

        Returns:
            (immunized_tensor, perturbation) both in [-1,1] half precision.
        """
        img_f = img.float()
        mask_f = img_mask.float()

        # Target: zero tensor (attack toward blank image)
        target = torch.zeros_like(img_f).cuda()

        # Prepare masked image (visible region only)
        X_adv = (img_f * (mask_f < 0.5).float()).clone().detach()

        pbar = tqdm(range(self.num_steps), desc="[PhotoGuard Diffusion]")
        for i in pbar:
            all_grads = []
            losses = []
            for _ in range(self.grad_reps):
                grad, loss_val = self._compute_grad(mask_f, X_adv, target)
                all_grads.append(grad)
                losses.append(loss_val)
            grad = torch.stack(all_grads).mean(0)

            pbar.set_description(
                f"[PhotoGuard Diffusion] AVG Loss: {np.mean(losses):.3f}"
            )

            # L2 normalized gradient step
            l = len(img_f.shape) - 1
            grad_norm = torch.norm(
                grad.detach().reshape(grad.shape[0], -1), dim=1
            ).view(-1, *([1] * l))
            grad_normalized = grad.detach() / (grad_norm + 1e-10)

            X_adv = X_adv - grad_normalized * self.step_size

            # Project onto L2 ball
            X_orig_masked = img_f * (mask_f < 0.5).float()
            d_x = X_adv - X_orig_masked
            d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=self.eps)
            X_adv = torch.clamp(
                X_orig_masked + d_x_norm, self.clamp_min, self.clamp_max
            )

            if i % 25 == 0:
                torch.cuda.empty_cache()

        # Reconstruct full image: adv outside mask + original inside mask
        immunized = X_adv.detach() * (1 - mask_f) + img_f * mask_f
        immunized = torch.clamp(immunized, self.clamp_min, self.clamp_max)
        perturbation = (immunized - img_f) * (1 - mask_f)

        del target
        torch.cuda.empty_cache()
        gc.collect()

        return immunized.half(), perturbation.half()

    def edit_image(self, prompt, img, img_mask, num_inf=30, SEED=5):
        """Edit image using the diffusion inpainting pipeline."""
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return self.attack_model.model(
            prompt=prompt, image=img, mask_image=img_mask,
            eta=1, num_inference_steps=num_inf,
            guidance_scale=7.5, strength=1.0, generator=generator,
        ).images
