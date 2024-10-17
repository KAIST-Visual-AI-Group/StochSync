from .camera_dataset import CameraDataset, RandomCameraDataset, RandomMVCameraDataset, SeqTurnaroundCameraDataset, FixMVCameraDataset, AlternateMVCameraDataset, AlternateCameraDataset
from .camera_dataset import QuatCameraDataset, TorusCameraDataset
from .image_dataset import ImageDataset, RotateImageDataset, RotateBatchImageDataset, ImageWideDataset, ImageWideRandomDataset, AlternateImageWideDataset
from .video_dataset import VideoDataset

DATASETs = {
    "camera": CameraDataset,
    "random": RandomCameraDataset,
    "random_mv": RandomMVCameraDataset,
    "seq_turnaround": SeqTurnaroundCameraDataset,
    "quat": QuatCameraDataset,
    "torus": TorusCameraDataset,
    "image": ImageDataset,
    "rotate_image": RotateImageDataset,
    "rotate_batch_image": RotateBatchImageDataset,
    "image_wide": ImageWideDataset,
    "image_wide_random": ImageWideRandomDataset,
    "fix_mv": FixMVCameraDataset,
    "alternate_mv": AlternateMVCameraDataset,
    "alternate_image_wide": AlternateImageWideDataset,
    "alternate_camera": AlternateCameraDataset,
    "video": VideoDataset,
}