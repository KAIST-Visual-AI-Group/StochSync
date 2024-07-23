from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class DistillationSampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, camera, images, step):
        # Calculate the score loss
        pass
