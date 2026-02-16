"""Clip Score Metric"""

import open_clip
import torch

from .base import Metric


class ClipScore(Metric):
    """CLIP score metric for evaluating the quality of an image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            kwargs["model"], pretrained=kwargs["pretrained_on"])
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(kwargs["model"])

    def __call__(self, edited_images, prompts):
        clip_scores = []
        for img, prompt in zip(edited_images, prompts):
            clip_score = self.calculate_clip_score(img, prompt)
            clip_scores.append(clip_score)
        return clip_scores

    def calculate_clip_score(self, img, prompt):
        """Calculate the CLIP score between an image and a prompt."""
        image = self.preprocess(img).unsqueeze(0)
        text = self.tokenizer([prompt])

        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_score = (100 * image_features @ text_features.T).mean()
        return clip_score.item()
