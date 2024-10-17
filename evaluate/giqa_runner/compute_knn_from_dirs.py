"""
compute_knn_from_dirs.py

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


WRITE_ACT_PATH = Path(
    "/home/dreamy1534/Projects/cvpr2024/code/diffmetahandles/tests/scripts/evaluation/giqa_runner/GIQA/write_act.py"
)
assert WRITE_ACT_PATH.exists(), (
    f"write_act.py not found at {WRITE_ACT_PATH}."
)

KNN_SCRIPT_PATH = Path(
    "/home/dreamy1534/Projects/cvpr2024/code/diffmetahandles/tests/scripts/evaluation/giqa_runner/GIQA/knn_score.py"
)
assert KNN_SCRIPT_PATH.exists(), (
    f"KNN script not found at {KNN_SCRIPT_PATH}."
)


@dataclass
class Args:
    fake_image_dirs: List[Path]
    """The image directories to be evaluated"""
    ref_image_dir: Path
    """The directory holding reference image files"""
    out_dir: Path
    """The output txt files recording the KNN scores"""


def main(args: Args):

    # parse args
    fake_image_dirs = args.fake_image_dirs
    ref_image_dir = args.ref_image_dir
    assert ref_image_dir.exists(), (
        f"Reference image directory {str(ref_image_dir)} does not exist"
    )
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created output directory: {str(out_dir)}")

    # compute inception features for reference images
    ref_actvn_file = out_dir / "ref_actvn.pkl"
    os.system(
        f"python {WRITE_ACT_PATH} {str(ref_image_dir)} --act_path={str(ref_actvn_file)} --gpu=0"
    )
    print("Computed reference features")
    assert ref_actvn_file.exists(), (
        f"Reference features not found at {ref_actvn_file}"
    )

    tmp_dir = out_dir / "tmp_results"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "summary.txt", mode="w") as f:

        for image_dir in tqdm(fake_image_dirs):
            assert image_dir.exists(), f"Image directory not found at {image_dir}"

            image_dir_name = image_dir.name
            out_file = tmp_dir / f"{image_dir_name}.txt"

            if not out_file.exists():
                os.system(
                    f"python {KNN_SCRIPT_PATH} {str(image_dir)} {str(out_file)} --act_path={str(ref_actvn_file)} --gpu=0"
                )
            else:
                print(
                    f"Output {str(out_file)} exists "
                    "Skipping evaluation"
                )

            names, scores, mean_score = compute_mean_score(out_file)

            f.write(f"{image_dir_name}: {mean_score}\n")

            ####
            # print the names of top 10 and worst 10 cases
            # TODO: save them as images
            scores_np = np.array(scores)
            scores_argsort = np.argsort(scores_np)

            top_10_indices = scores_argsort[-10:].tolist()
            worst_10_indices = scores_argsort[:10].tolist()

            top_10_names = [names[i] for i in top_10_indices]
            worst_10_names = [names[i] for i in worst_10_indices]
            print(
                f"{image_dir_name} Top 10: {top_10_names}"
            )
            print(
                f"{image_dir_name} Worst 10: {worst_10_names}"
            )
            top_10_images = [read_image(image_dir / name) for name in top_10_names]
            top_10_images = np.stack(top_10_images, axis=0)
            worst_10_images = [read_image(image_dir / name) for name in worst_10_names]
            worst_10_images = np.stack(worst_10_images, axis=0)

            top_10_images = make_image_grid(top_10_images, ncol=5)
            worst_10_images = make_image_grid(worst_10_images, ncol=5)

            write_image(
                top_10_images,
                out_dir / f"{image_dir_name}_top_10.png",
            )
            write_image(
                worst_10_images,
                out_dir / f"{image_dir_name}_worst_10.png",
            )
            ####
            

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
