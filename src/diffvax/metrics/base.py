"""Base class for metrics."""


class Metric:
    """Base class for metrics."""

    name: str

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
