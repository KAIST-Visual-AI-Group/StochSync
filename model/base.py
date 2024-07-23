import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseModel(ABC):
    """
    Abstract base class for all 3D models. This class defines the interface that
    all 3D models must implement.
    """

    def __init__(self) -> None:
        pass

    @property
    def device(self) -> str:
        """
        Returns the device (cpu or cuda) to be used for computations.
        
        :return: Device to be used (cpu or cuda).
        :raises AssertionError: If the configuration is not initialized.
        """
        assert self.cfg is not None, "Configuration not initialized"
        return self.cfg.device

    @property
    def max_steps(self) -> int:
        """
        Returns the maximum number of steps for optimization.
        
        :return: Maximum number of steps.
        :raises AssertionError: If the configuration is not initialized.
        """
        assert self.cfg is not None, "Configuration not initialized"
        return self.cfg.max_steps

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        
        :param path: Path to the file from which to load the model.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        :param path: Path to the file to which to save the model.
        """
        pass

    @abstractmethod
    def prepare_optimization(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare the model for optimization. This might include setting up optimizers and other related tasks.
        
        :param parameters: Optional dictionary of parameters for optimization.
        """
        pass

    @abstractmethod
    def render(self, c2ws: torch.Tensor, Ks: torch.Tensor, width: int, height: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Render the 3D model from the specified camera parameters.
        
        :param c2ws: Camera-to-world transformation matrices.
        :param Ks: Camera intrinsic matrices.
        :param width: Width of the rendered image.
        :param height: Height of the rendered image.
        :param kwargs: Additional parameters for rendering.
        :return: A dictionary containing the rendered images and other information.
        """
        pass

    @abstractmethod
    def optimize(self, step: int) -> None:
        """
        Optimize the model parameters for the given step.
        
        :param step: Current step of the optimization process.
        """
        pass

    def regularize(self):
        """
        Regularize the model parameters.
        """
        return torch.tensor(0.0, device=self.device)