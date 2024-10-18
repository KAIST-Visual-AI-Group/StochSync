import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import tempfile

import yaml

# file path
FILE_DIR = Path(__file__).resolve().parent
WRITE_ACT_PATH = FILE_DIR / "giqa_runner/GIQA/write_act.py"
assert WRITE_ACT_PATH.exists(), f"write_act.py not found at {WRITE_ACT_PATH}."

KNN_SCRIPT_PATH = FILE_DIR / "giqa_runner/GIQA/knn_score.py"
assert KNN_SCRIPT_PATH.exists(), f"knn_score.py not found at {KNN_SCRIPT_PATH}."


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


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


class GIQAEvaluator:
    def __init__(
        self,
    ):
        pass

    def extract_features(
        self,
        image_dir: Union[str, os.PathLike],
        act_path: Union[str, os.PathLike],
        gpu_id: int,
    ):
        cmd = (
            f"python {WRITE_ACT_PATH} {image_dir} --act_path {act_path} --gpu {gpu_id}"
        )
        os.system(cmd)

    def compute_knn_score(
        self,
        image_dir: Union[str, os.PathLike],
        out_path: Union[str, os.PathLike],
        ref_act_path: Union[str, os.PathLike],
        gpu_id: int,
    ):
        assert Path(
            image_dir
        ).exists(), f"Test image directory not found at {image_dir}"
        assert Path(
            ref_act_path
        ).exists(), f"Reference activation not found at {ref_act_path}"

        cmd = f"python {KNN_SCRIPT_PATH} {image_dir} --output_file {out_path} --act_path {ref_act_path} --gpu {gpu_id}"
        print(cmd)
        os.system(cmd)

    def __call__(
        self,
        ref_image_dir: Union[str, os.PathLike],
        fake_image_dir: Union[str, os.PathLike],
        feature_dir: Optional[Union[str, os.PathLike]] = None,
        gpu_id: Optional[int] = 0,
    ):

        # Path(save_dir).mkdir(exist_ok=True, parents=True)
        if feature_dir is not None:
            os.makedirs(feature_dir, exist_ok=True)
        else:
            feature_dir = tempfile.TemporaryDirectory()

        ref_act_path = Path(feature_dir) / f"ref_feature.pkl"
        if not os.path.exists(ref_act_path):
            self.extract_features(ref_image_dir, ref_act_path, gpu_id)
            print(f"[*] Computed reference image features")

        with tempfile.TemporaryDirectory() as temp_dir:
            now = get_current_time()
            output_file_path = Path(temp_dir) / f"all_knn_scores_method_{now}.txt"
            self.compute_knn_score(
                fake_image_dir, output_file_path, ref_act_path=ref_act_path, gpu_id=gpu_id
            )

            names, scores, mean_score = compute_mean_score(output_file_path)

        return mean_score
        dic = {
            "ref_image_dir": ref_image_dir,
            "fake_image_dir": fake_image_dir,
            "giqa_knn_score": float(mean_score),
        }

        yaml_path = Path(save_dir) / f"giqa_method_{self.method}_{now}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dic, f)

        print(f"[*] Compared {ref_image_dir}(ref) and {fake_image_dir}(fake) | GIQA: {mean_score}")
        print(f"[*] Logged at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image_dir", type=str, required=True)
    parser.add_argument("--fake_image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--exist_ref_act_path", type=str, default=None)

    args = parser.parse_args()

    giqa_evaluator = GIQAEvaluator(
        args.ref_image_dir, 
        args.fake_image_dir, 
        args.save_dir, 
        args.gpu_id, 
        args.method, 
        args.exist_ref_act_path,
    )

    # If you already have a pre-computed reference feature data, you can load it and skip the computation by passing `exist_ref_act_path`.
    giqa_evaluator(args.ref_image_dir, args.fake_image_dir, args.save_dir, args.gpu_id)