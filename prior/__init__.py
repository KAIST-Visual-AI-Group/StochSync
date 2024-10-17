from .sd import (
    StableDiffusionPrior,
    AngleDependentStableDiffusionPrior,
    ViewDependentStableDiffusionPrior,
    ElevDependentStableDiffusionPrior,
    CubemapStableDiffusionPrior,
    UltimateStableDiffusionPrior
)
from .inpainting import InpaintingPrior
from .controlnet import ControlNetPrior, SD2DepthPrior
from .mvdream import MVDreamPrior

from .deepfloyd import DeepFloydPrior

PRIORs = {
    "sd": StableDiffusionPrior,
    "view_sd": ViewDependentStableDiffusionPrior,
    "elev_sd": ElevDependentStableDiffusionPrior,
    "angle_sd": AngleDependentStableDiffusionPrior,
    "ultimate_sd": UltimateStableDiffusionPrior,
    "cubemap_sd": CubemapStableDiffusionPrior,
    "inpainting": InpaintingPrior,
    "sd2_depth": SD2DepthPrior,
    "controlnet": ControlNetPrior,
    "mvdream": MVDreamPrior,
    "df": DeepFloydPrior,
}
