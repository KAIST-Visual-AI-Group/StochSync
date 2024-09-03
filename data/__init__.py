from .camera_dataset import CameraDataset, RandomCameraDataset, RandomMVCameraDataset, SeqTurnaroundCameraDataset#, NVDiffrastCameraDataset, NVDiffrastMVDreamCameraDataset
from .image_dataset import ImageDataset, RotateImageDataset, RotateBatchImageDataset, ImageWideDataset, ImageWideRandomDataset

DATASETs = {
    "camera": CameraDataset,
    "random": RandomCameraDataset,
    "random_mv": RandomMVCameraDataset,
    "seq_turnaround": SeqTurnaroundCameraDataset,
    # "nv_diffrast": NVDiffrastCameraDataset,
    # "nv_diffrast_mv": NVDiffrastMVDreamCameraDataset,
    "image": ImageDataset,
    "rotate_image": RotateImageDataset,
    "rotate_batch_image": RotateBatchImageDataset,
    "image_wide": ImageWideDataset,
    "image_wide_random": ImageWideRandomDataset,
}