from .gs import GSModel
from .mesh import MeshModel, PaintitMeshModel
from .image import ImageModel
from .image_mv import ImageMVModel
from .image_wide import ImageWideModel
# from .ngp import NGPModel
from .panorama import PanoramaModel

MODELs = {
    "gs": GSModel,
    "mesh": MeshModel,
    "paintit_mesh": PaintitMeshModel,
    "image": ImageModel,
    "image_mv": ImageMVModel,
    "image_wide": ImageWideModel,
    # "ngp": NGPModel,
    "panorama": PanoramaModel,
}