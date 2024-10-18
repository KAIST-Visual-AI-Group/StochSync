from .gs import GSModel
from .mesh import MeshModel
from .image import ImageModel
from .image_wide import ImageWideModel
from .ngp import NGPModel
from .panorama import PanoramaModel

MODELs = {
    "gs": GSModel,
    "mesh": MeshModel,
    "image": ImageModel,
    "image_wide": ImageWideModel,
    "ngp": NGPModel,
    "panorama": PanoramaModel,
}