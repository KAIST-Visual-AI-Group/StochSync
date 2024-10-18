# import cv2
import torch
import os
import numpy as np
from third_party.perspective2d import PerspectiveFields
from PIL import Image
from utils.image_utils import torch_to_pil, pil_to_torch
from torchvision.transforms import functional as F
import cv2

PerspectiveFields.versions()
pf_model = PerspectiveFields('Paramnet-360Cities-edina-centered').eval().cuda()

from utils.path_utils import gather_paths, filter_paths, collect_keys
import tqdm


path = "/home/jh27kim/wkst/docker_home/code/current_projects/iclr2025/DistillAnywhere/results/baselines/horizon/LMAGIC_raw/pano"
pattern = f"{path}/:0:_:1:_:2:.png"
path_dict = gather_paths(pattern)
prompts = collect_keys(path_dict, 0)
total_roll_error = 0
total_pitch_error = 0
total_fov_error = 0
for prompt in tqdm.tqdm(prompts):
    paths = filter_paths(path_dict, prompt)
    paths = paths.values()
    # load images
    images = [cv2.imread(p) for p in paths]
    predictions = pf_model.inference_batch(images)
    rolls = []
    pitches = []
    fovs = []
    for pred in predictions:
        rolls.append(pred['pred_roll'].item())
        pitches.append(pred['pred_pitch'].item())
        fovs.append(pred["pred_general_vfov"].item())
    rolls = torch.tensor(rolls)
    pitches = torch.tensor(pitches)
    fovs = torch.tensor(fovs)
    roll_error = torch.mean(rolls.abs())
    pitch_error = torch.mean((pitches - torch.mean(pitches)).abs())
    fov_error = torch.mean((fovs - 72).abs())
    tqdm.tqdm.write(f"{prompt}: roll_error={roll_error:.3f}, pitch_error={pitch_error:.3f}, fov_error={fov_error:.3f}")
    total_roll_error += roll_error
    total_pitch_error += pitch_error
    total_fov_error += fov_error

total_roll_error /= len(prompts)
total_pitch_error /= len(prompts)
total_fov_error /= len(prompts)
print('*'*20)
print(path)
print(f"Total: roll_error={total_roll_error:.3f}, pitch_error={total_pitch_error:.3f}, fov_error={total_fov_error:.3f}")