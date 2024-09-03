from .sd import (
    StableDiffusionPrior,
    AngleDependentStableDiffusionPrior,
    ViewDependentStableDiffusionPrior,
    ElevDependentStableDiffusionPrior,
)
from .mvdream import MVDreamPrior

from .deepfloyd import DeepFloydPrior

PRIORs = {
    "sd": StableDiffusionPrior,
    "view_sd": ViewDependentStableDiffusionPrior,
    "elev_sd": ElevDependentStableDiffusionPrior,
    "angle_sd": AngleDependentStableDiffusionPrior,
    "mvdream": MVDreamPrior,
    "df": DeepFloydPrior,
}
