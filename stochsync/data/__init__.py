from .camera_dataset import CameraDataset, RandomCameraDataset, RandomMVCameraDataset, SeqTurnaroundCameraDataset, FixMVCameraDataset, AlternateCameraDataset
from .camera_dataset import QuatCameraDataset, TorusCameraDataset
from .image_dataset import ImageDataset, ImageWideDataset, RandomImageWideDataset, AlternateImageWideDataset, SixViewNoOverlapCameraDataset

DATASETs = {
    "camera": CameraDataset,
    "random": RandomCameraDataset,
    "random_mv": RandomMVCameraDataset,
    "seq_turnaround": SeqTurnaroundCameraDataset,
    "quat": QuatCameraDataset,
    "torus": TorusCameraDataset,
    "image": ImageDataset,
    "image_wide": ImageWideDataset,
    "image_wide_random": RandomImageWideDataset,
    "fix_mv": FixMVCameraDataset,
    # "alternate_mv": AlternateMVCameraDataset,
    "alternate_image_wide": AlternateImageWideDataset,
    "alternate_camera": AlternateCameraDataset,
    "vigilance_test": SixViewNoOverlapCameraDataset,
}