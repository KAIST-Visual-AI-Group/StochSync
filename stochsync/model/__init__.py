from .gs import GSModel
from .gs_texture import GSTextureModel
from .mesh import MeshModel
from .image import ImageModel
from .image_wide import ImageWideModel
from .image_inpainting import ImageInpaintingModel
from .ngp import NGPModel
from .panorama import PanoramaModel

MODELs = {
    "gs": GSModel,
    "gs_texture": GSTextureModel,
    "mesh": MeshModel,
    "image": ImageModel,
    "image_wide": ImageWideModel,
    "image_inpainting": ImageInpaintingModel,
    "ngp": NGPModel,
    "panorama": PanoramaModel,
}