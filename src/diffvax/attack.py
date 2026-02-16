from PIL import Image
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
)
from typing import Union, List, Optional, Callable


class Attack:
    def __init__(self, model_link: str, scheduler: str = "DDIM"):
        pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            model_link,
            torch_dtype=torch.float16,
        )
        if scheduler == "DDIM":
            pipe_inpaint.scheduler = DDIMScheduler.from_config(
                pipe_inpaint.scheduler.config
            )
        elif scheduler == "PNDM":
            pipe_inpaint.scheduler = PNDMScheduler.from_config(
                pipe_inpaint.scheduler.config
            )
        elif scheduler == "euler":
            pipe_inpaint.scheduler = EulerDiscreteScheduler.from_config(
                pipe_inpaint.scheduler.config
            )
        elif scheduler == "LMS":
            pipe_inpaint.scheduler = LMSDiscreteScheduler.from_config(
                pipe_inpaint.scheduler.config
            )
        elif scheduler == "KDPM":
            pipe_inpaint.scheduler = KDPM2DiscreteScheduler.from_config(
                pipe_inpaint.scheduler.config
            )

        self.model = pipe_inpaint.to("cuda")
        self.model_link = model_link

    def attack(
        self,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        batch_size: int = 1,
    ):
        """Differentiable forward pass of the inpainting stable diffusion model."""
        diffusion_model = self.model

        text_embeddings = self.tokenize_prompt(
            diffusion_model, prompt, batch_size=batch_size
        )

        num_channels_latents = diffusion_model.vae.config.latent_channels

        latents_shape = (
            batch_size,
            num_channels_latents,
            height // 8,
            width // 8,
        )
        latents = torch.randn(
            latents_shape,
            device=diffusion_model.device,
            dtype=text_embeddings.dtype,
        )

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)

        masked_image_latents = diffusion_model.vae.encode(
            masked_image
        ).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        latents = latents * diffusion_model.scheduler.init_noise_sigma

        diffusion_model.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = diffusion_model.scheduler.timesteps.to(
            diffusion_model.device
        )

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latents], dim=1
            )
            noise_pred = diffusion_model.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = diffusion_model.scheduler.step(
                noise_pred, t, latents, eta=eta
            ).prev_sample

        latents = 1 / 0.18215 * latents
        image = diffusion_model.vae.decode(latents).sample
        return image

    def tokenize_prompt(
        self, diffusion_model, prompt, batch_size=1, tokenize_negative=False
    ):
        """Tokenize prompts. Uses 'gray background' as unconditional embedding if tokenize_negative is True."""
        text_inputs = diffusion_model.tokenizer(
            prompt,
            padding="max_length",
            max_length=diffusion_model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = diffusion_model.text_encoder(
            text_input_ids.to(diffusion_model.device)
        )[0]

        uncond_tokens = [""] * batch_size
        if tokenize_negative:
            uncond_tokens = ["gray background"]
        max_length = text_input_ids.shape[-1]
        uncond_input = diffusion_model.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = diffusion_model.text_encoder(
            uncond_input.input_ids.to(diffusion_model.device)
        )[0]
        seq_len = uncond_embeddings.shape[1]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        text_embeddings = text_embeddings.detach()
        return text_embeddings
