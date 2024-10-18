from .gs import GSModel
from .mesh import MeshModel
from .image import ImageModel
# from .image_mv import ImageMVModel
from .image_wide import ImageWideModel
from .ngp import NGPModel
from .panorama import PanoramaModel
# from .cubemap import CubemapModel
from .video import VideoModel

MODELs = {
    "gs": GSModel,
    "mesh": MeshModel,
    "image": ImageModel,
    # "image_mv": ImageMVModel,
    "image_wide": ImageWideModel,
    "ngp": NGPModel,
    "panorama": PanoramaModel,
    # "cubemap": CubemapModel,
    "video": VideoModel,
}