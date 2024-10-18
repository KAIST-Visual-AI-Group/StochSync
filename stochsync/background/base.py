import torch
from abc import ABC, abstractmethod


class BaseBackground(ABC):
    """
    A simple background class for generating background images.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass