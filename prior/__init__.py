from .sd import (
    StableDiffusionPrior,
    AngleDependentStableDiffusionPrior,
    ViewDependentStableDiffusionPrior,
    ElevDependentStableDiffusionPrior,
)
from .controlnet import ControlNetPrior
from .mvdream import MVDreamPrior

from .deepfloyd import DeepFloydPrior

PRIORs = {
    "sd": StableDiffusionPrior,
    "view_sd": ViewDependentStableDiffusionPrior,
    "elev_sd": ElevDependentStableDiffusionPrior,
    "angle_sd": AngleDependentStableDiffusionPrior,
    "controlnet": ControlNetPrior,
    "mvdream": MVDreamPrior,
    "df": DeepFloydPrior,
}
