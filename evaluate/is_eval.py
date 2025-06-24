import os
import yaml 
import argparse
import glob 
from datetime import datetime
from pathlib import Path
from typing import Literal, Union
from PIL import Image 
import torch 
import numpy as np

from torchmetrics.image.inception import InceptionScore

def pil_to_torch(pil_img):
    _np_img = np.array(pil_img)
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


class ISEvaluator:
    def __init__(
        self,
        fdir1: Union[str, os.PathLike] = None,
        save_dir: Union[str, os.PathLike] = None,
        method: str = None,
    ):
        self.fdir1 = fdir1
        self.InceptionScore = InceptionScore()
        self.InceptionScore.to(device="cuda")
        self.device = self.InceptionScore.device

        self.save_dir = save_dir
        self.method = method

    def __call__(self, fdir1=None):
        if fdir1 is None:
            fdir1 = self.fdir1

        assert Path(fdir1).exists(), f"{fdir1} not exist."

        for img_path in glob.glob(f"{fdir1}/*"):
            img = Image.open(img_path).convert("RGB")
            torch_img = pil_to_torch(img).to(self.device)

            self.InceptionScore.update(torch_img)

        inception_mean, inception_std = self.InceptionScore.compute()
        return inception_mean.item()

        dic = {
            "fdir1": fdir1,
            metric_name: inception_mean.item(),
            f"{metric_name}_std": inception_std.item(),
        }

        Path(save_dir).mkdir(exist_ok=True)
        now = get_current_time()
        yaml_path = Path(save_dir) / f"{metric_name}_method_{self.method}_{now}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dic, f)

        print(f"[*] Generated images {fdir1} | {metric_name}: mean: {inception_mean} | std: {inception_std}")
        print(f"[*] Logged at {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir1", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--metric", type=str, default="is", choices=["is"])
    parser.add_argument("--method", type=str, required=True)

    args = parser.parse_args()

    fdir1 = args.fdir1  # imgae directory 1
    save_dir = args.save_dir  # log directory
    method = args.method
    is_evaluator = ISEvaluator(fdir1, save_dir, method)

    is_evaluator(fdir1)
