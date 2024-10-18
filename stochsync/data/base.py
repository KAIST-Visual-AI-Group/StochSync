import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod


class InfiniteDataset(IterableDataset, ABC):
    """
    A simple abstract class for infinitely large datasets.
    It provides an iterator that generates data on-the-fly.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_sample(self) -> torch.Tensor:
        pass

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        return self.generate_sample()