"""
compute_knn_per_cls.py

A script for computing KNN GIQA score from multiple experiment result directory.
"""


from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import tyro

from src.utils.io.image.imageio import (
    read_image,
    write_image,
)
from src.utils.io.image.ops import (
    make_image_grid
)

KNN_SCRIPT_PATH = Path(
    "/home/dreamy1534/Projects/cvpr2024/code/diffmetahandles/tests/scripts/evaluation/giqa_runner/GIQA/knn_score.py"
)
assert KNN_SCRIPT_PATH.exists(), (
    f"KNN script not found at {KNN_SCRIPT_PATH}."
)

@dataclass
class Args:
    img_root_dir: Path
    """The image directories to be evaluated"""
    ref_actvn_dir: Path
    """The directory holding reference activation files"""
    out_dir: Path
    """The output txt files recording the KNN scores"""


def main(args: Args):

    # parse args
    img_dirs = list(args.img_root_dir.iterdir())
    for img_dir in img_dirs:
        assert img_dir.exists(), (
            f"Image directory {str(img_dir)} does not exist"
        )
    ref_actvn_dir = args.ref_actvn_dir
    assert ref_actvn_dir.exists(), (
        f"Reference activation directory {str(ref_actvn_dir)} does not exist"
    )
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"Created output directory: {str(out_dir)}"
    )
    tmp_dir = out_dir / "knn_tmp_results"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "summary.txt", mode="w") as f:
        for img_dir in tqdm(img_dirs):    

            exp_name = img_dir.name

            # evaluate KNN score
            (
                total_mean_score,
                per_cls_mean_score,
            ) = compute_knn_score_per_exp(
                img_dir,
                ref_actvn_dir,
                tmp_dir / exp_name,        
            )

            print(f"{exp_name}: {total_mean_score:.5f}")

            # log
            f.write(f"{exp_name}: {total_mean_score:.5f}\n")

def compute_knn_score_per_exp(
    img_dir: Path,
    ref_actvn_dir: Path,
    tmp_dir: Path,
) -> float:
    tmp_dir.mkdir(exist_ok=True, parents=True)

    per_cls_mean_score = {}
    total_mean_score = 0.0

    for cls_dir in img_dir.iterdir():
        cls_name = cls_dir.name

        # TODO: look up the reference features
        ref_actvn_file = ref_actvn_dir / f"{cls_name}_actvn.pkl"
        assert ref_actvn_file.exists(), (
            f"Reference features not found at {ref_actvn_file}"
        )

        # TODO: Run knn_score.py for the current class
        out_file = tmp_dir / f"{cls_name}.txt"
        if not out_file.exists():
            os.system(
                f"python {KNN_SCRIPT_PATH} {str(cls_dir)} {str(out_file)} --act_path={str(ref_actvn_file)} --K=12 --gpu=0"
            )            
        else:
            print(
                f"Output {str(out_file)} exists "
                "Skipping evaluation"
            )
        names, scores, mean_score = compute_mean_score(out_file)
        
        # collect result
        per_cls_mean_score[str(cls_name)] = mean_score
        total_mean_score += mean_score
    total_mean_score /= len(per_cls_mean_score)

    return total_mean_score, per_cls_mean_score


def compute_mean_score(file) -> float:
    
    with open(file, mode="r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    # parse file names
    names = lines[0::2]
    names = [line.split(" ")[2] for line in names]

    # parse scores
    scores = lines[1::2]
    scores = [float(score) for score in scores]
    mean_score = sum(scores) / len(scores)
    return names, scores, mean_score


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
