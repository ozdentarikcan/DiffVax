"""SSIM metric for evaluating the quality of an image."""

from .base import Metric
from skimage.metrics import structural_similarity

class SSIM(Metric):

    def __call__(self, original_images, adversarial_images):
        """Calculate the SSIM between original and adversarial images."""
        ssim_values = []
        for img_orig, img_adv in zip(original_images, adversarial_images):
            ssim = self.calculate_metric_between_images(img_orig, img_adv)
            ssim_values.append(ssim)
        return ssim_values
    
    def calculate_metric_between_images(self, img_orig, img_adv):
        """Calculate the SSIM between two images."""
        ssim = structural_similarity(img_orig, img_adv, channel_axis=2)
        return ssim