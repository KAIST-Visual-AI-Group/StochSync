import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.io import read_video, write_video
import torch.nn.functional as TF


def forward_warping(src_img, optical_flow, return_mask=False):
    """
    Warps the src_img using the optical_flow (forward warping).
    The optical flow describes how to move from the source frame to the destination frame.

    Parameters:
    src_img (torch.Tensor): [1, C, H, W] The source image (tensor of shape (1, channels, height, width))
    optical_flow (torch.Tensor): [1, 2, H, W] Optical flow (shape (1, 2, height, width))

    Returns:
    torch.Tensor: Warped image of shape (1, C, H, W)
    """
    B, C, H, W = src_img.shape
    device = src_img.device

    # Generate a mesh grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid_x = grid_x.to(device).float()
    grid_y = grid_y.to(device).float()

    # Apply the optical flow to find destination pixel locations
    flow_x = grid_x + optical_flow[:, 0, :, :]
    flow_y = grid_y + optical_flow[:, 1, :, :]

    # Round to nearest pixel values for forward warping
    flow_x = flow_x.clamp(0, W - 1).round().long()
    flow_y = flow_y.clamp(0, H - 1).round().long()

    # Create an empty destination image
    warped_img = torch.zeros_like(src_img)
    warped_mask = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)

    # For each pixel in the source image, place its value in the destination image
    for c in range(C):
        warped_img[0, c, flow_y, flow_x] = src_img[0, c, grid_y.long(), grid_x.long()]
        warped_mask[0, 0, flow_y, flow_x] = 1

    if return_mask:
        return warped_img, warped_mask
    return warped_img


def backward_warping(src_img, optical_flow, return_mask=False):
    """
    Warps the src_img using the optical_flow (backward warping).
    The optical flow describes how to map from the destination frame back to the source frame.

    Parameters:
    src_img (torch.Tensor): [1, C, H, W] The source image (tensor of shape (1, channels, height, width))
    optical_flow (torch.Tensor): [1, 2, H, W] Optical flow (shape (1, 2, height, width))

    Returns:
    torch.Tensor: Warped image of shape (1, C, H, W)
    """
    B, C, H, W = src_img.shape
    device = src_img.device

    # Generate a mesh grid of coordinates (normalized to [-1, 1] for grid_sample)
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid_x = grid_x.to(device).float() / (W - 1) * 2 - 1  # Normalize to [-1, 1]
    grid_y = grid_y.to(device).float() / (H - 1) * 2 - 1  # Normalize to [-1, 1]

    # Apply the flow to the grid
    flow_x = optical_flow[:, 0, :, :] / W  # Flow in x direction normalized
    flow_y = optical_flow[:, 1, :, :] / H  # Flow in y direction normalized

    # Create the grid for sampling by adding flow to the grid
    flow_grid = torch.stack((grid_x + flow_x, grid_y + flow_y), dim=-1)

    # Perform grid sampling (this does the backward warping)
    warped_img = TF.grid_sample(src_img, flow_grid, mode="nearest", align_corners=False)
    warped_mask = TF.grid_sample(
        torch.ones((1, 1, H, W), dtype=torch.float32, device=device),
        flow_grid,
        mode="nearest",
        align_corners=False,
    )

    if return_mask:
        return warped_img, warped_mask
    return warped_img


def get_flow_estimator(device="cuda"):
    return raft_large(pretrained=True, progress=False).to(device)


def get_optical_flow_raw(
    src_frames, dest_fraces, batch_size=8, resize=None, model=None
):
    def preprocess(batch, resize=None):
        if resize:
            r = [T.Resize(size=(resize, resize))]
        else:
            r = []
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            ]
            + r
        )
        batch = transforms(batch)
        return batch

    device = src_frames.device
    if model is None:
        model = raft_large(pretrained=True, progress=False).to(device)
        model = model.eval()

    if src_frames.shape[0] == 1:
        src_frames = src_frames.repeat(dest_fraces.shape[0], 1, 1, 1)

    assert src_frames.shape == dest_fraces.shape

    B, C, H, W = src_frames.shape
    if resize:
        H, W = resize, resize

    batch_repeat = (len(src_frames) - 1) // batch_size + 1
    # print(f"{batch_repeat=}")
    optical_flow = torch.zeros(B, 2, H, W, device=device)

    # total_flows = []
    for i in range(batch_repeat):
        size = min(batch_size, B - i * batch_size)

        img1_batch = src_frames[i * batch_size : i * batch_size + size]
        img2_batch = dest_fraces[i * batch_size : i * batch_size + size]

        img1_batch = preprocess(img1_batch, resize).to(device)
        img2_batch = preprocess(img2_batch, resize).to(device)

        with torch.no_grad():
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            optical_flow[i * batch_size : i * batch_size + size, ...] = list_of_flows[
                -1
            ].cpu()

    return optical_flow


def get_optical_flow(videodir, batch_size=8, resize=None, save_dir=None, prev=False):
    """
    params:
        videodir: String - path to the video file
        batch_size: Int - number of frames to process in a batch
        resize: Int or None - optional resizing to square dimensions (resize, resize)
        save_dir: String - directory where the results will be saved
        prev: Bool - whether to use the previous frame as the anchor or the first frame
    return:
        optical_flow [B, 2, H, W] (B: frame#)
    """

    def preprocess(batch, resize=None):
        if resize:
            r = [T.Resize(size=(resize, resize))]
        else:
            r = []
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            ]
            + r
        )
        batch = transforms(batch)
        return batch

    frames, _, _ = read_video(videodir)
    write_video(f"{save_dir}/original.mp4", frames, fps=15)

    frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    # frames = frames[:32,...]  # Limiting to the first 32 frames for processing
    B, C, H, W = frames.shape
    if resize:
        H, W = resize, resize

    batch_repeat = (len(frames) - 1) // batch_size + 1
    print(f"{batch_repeat=}")
    optical_flow = torch.zeros(B, 2, H, W)

    anchor = torch.cat([frames[0:1], frames[:-1]])

    if prev:
        optical_flow = get_optical_flow_raw(
            anchor, frames, batch_size=batch_size, resize=resize, save_dir=save_dir
        )
    else:
        optical_flow = get_optical_flow_raw(
            frames[0:1].repeat(B, 1, 1, 1),
            frames,
            batch_size=batch_size,
            resize=resize,
            save_dir=save_dir,
        )

    total_flows = flow_to_image(optical_flow).permute(0, 2, 3, 1).cpu()
    print(total_flows.shape)
    os.makedirs(f"{save_dir}/optical_flow", exist_ok=True)
    write_video(f"{save_dir}/optical_flow.mp4", total_flows, fps=15)

    return optical_flow


if __name__ == "__main__":
    video_path = "video/puff.mp4"
    save_directory = "./_temp/"
    flow = get_optical_flow(
        videodir=video_path,
        batch_size=8,
        resize=256,
        save_dir=save_directory,
        prev=True,
    )
