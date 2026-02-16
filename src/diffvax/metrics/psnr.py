"""PSNR metric for evaluating the quality of an image."""

from .base import Metric
from skimage.metrics import peak_signal_noise_ratio


class PSNR(Metric):
    """Peak Signal to Noise Ratio (PSNR) metric for evaluating the quality of an image."""

    def __call__(self, original_images, adversarial_images):
        """Calculate the PSNR between original and adversarial images."""
        psnr_values = []
        for img_orig, img_adv in zip(original_images, adversarial_images):
            psnr = self.calculate_metric_between_images(img_orig, img_adv)
            psnr_values.append(psnr)
        return psnr_values
    
    def calculate_metric_between_images(self, img_orig, img_adv):
        """Calculate the PSNR between two images."""
        psnr = peak_signal_noise_ratio(img_orig, img_adv)
        return psnr
