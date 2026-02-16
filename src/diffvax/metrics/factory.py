"""Metric Factory Function for creating metrics."""

from enum import Enum
from .psnr import PSNR
from .ssim import SSIM
from .clip_score import ClipScore
from .fsim import FSIM


class MetricType(Enum):
    """Enum for the metrics."""
    PSNR = "PSNR"
    SSIM = "SSIM"
    CLIP = "CLIP"
    FSIM = "FSIM"


def create_metric(metric_type: MetricType, **kwargs):
    """Factory function for creating metrics."""
    if metric_type == MetricType.PSNR:
        return PSNR()
    elif metric_type == MetricType.SSIM:
        return SSIM()
    elif metric_type == MetricType.CLIP:
        return ClipScore(**kwargs)
    elif metric_type == MetricType.FSIM:
        return FSIM()
    else:
        raise ValueError(f"Invalid metric name: {metric_type}")
