from .camera_dataset import CameraDataset, RandomCameraDataset, RandomMVCameraDataset, SeqTurnaroundCameraDataset#, NVDiffrastCameraDataset, NVDiffrastMVDreamCameraDataset
from .image_dataset import ImageDataset, RotateImageDataset, RotateBatchImageDataset, ImageWideDataset, ImageWideRandomDataset, TwoViewNoOverlapCameraDataset, TwoViewOverlapCameraDataset, TwoViewOverlapCameraDataset

DATASETs = {
    "camera": CameraDataset,
    "random": RandomCameraDataset,
    "random_mv": RandomMVCameraDataset,
    "seq_turnaround": SeqTurnaroundCameraDataset,
    "image": ImageDataset,
    "rotate_image": RotateImageDataset,
    "rotate_batch_image": RotateBatchImageDataset,
    "image_wide": ImageWideDataset,
    "image_wide_random": ImageWideRandomDataset,
    # 2-view dataset for debugging
    "two_view_no_overlap": TwoViewNoOverlapCameraDataset,
    "two_view_overlap": TwoViewOverlapCameraDataset,
}