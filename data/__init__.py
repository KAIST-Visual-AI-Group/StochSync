from .camera_dataset import CameraDataset, RandomCameraDataset, RandomMVCameraDataset, SeqTurnaroundCameraDataset, FixMVCameraDataset, AlternateMVCameraDataset
from .image_dataset import ImageDataset, RotateImageDataset, RotateBatchImageDataset, ImageWideDataset, ImageWideRandomDataset, AlternateImageWideDataset

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
    "fix_mv": FixMVCameraDataset,
    "alternate_mv": AlternateMVCameraDataset,
    "alternate_image_wide": AlternateImageWideDataset,
}