from .gs import GSModel
from .gs_reg import GSModelReg
from .mesh import MeshModel, PaintitMeshModel
from .image import ImageModel, LatentImageModel

MODELs = {
    "gs": GSModel,
    "gs_reg": GSModelReg,
    "mesh": MeshModel,
    "paintit_mesh": PaintitMeshModel,
    "image": ImageModel,
    "latent_image": LatentImageModel,
}