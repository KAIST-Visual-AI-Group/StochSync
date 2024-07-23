from .sd import StableDiffusionPrior, AngleDependentStableDiffusionPrior, MVDreamPrior
from .deepfloyd import DeepFloydPrior

PRIORs = {
    "sd": StableDiffusionPrior,
    "angle_sd": AngleDependentStableDiffusionPrior,
    "df": StableDiffusionPrior,
    "mvdream": MVDreamPrior,
}