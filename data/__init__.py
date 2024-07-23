from .camera_dataset import RandomCameraDataset, RandomMVDreamCameraDataset, SeqTurnaroundCameraDataset#, NVDiffrastCameraDataset, NVDiffrastMVDreamCameraDataset
from .image_dataset import ImageDataset, RotateImageDataset, RotateBatchImageDataset

DATASETs = {
    "random": RandomCameraDataset,
    "random_mv": RandomMVDreamCameraDataset,
    "seq_turnaround": SeqTurnaroundCameraDataset,
    # "nv_diffrast": NVDiffrastCameraDataset,
    # "nv_diffrast_mv": NVDiffrastMVDreamCameraDataset,
    "image": ImageDataset,
    "rotate_image": RotateImageDataset,
    "rotate_batch_image": RotateBatchImageDataset,
}