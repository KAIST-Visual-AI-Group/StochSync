import os
import sys

# add the parent folder to the python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import argparse
import csv
from itertools import product
import tempfile
import shutil
from cleanfid import fid
from clip_eval import ClipEvaluator
from giqa_eval import GIQAEvaluator
from utils.print_utils import print_info, print_error, print_warning, print_with_box
from utils.path_utils import gather_paths, filter_paths, collect_keys
from natsort import natsorted


def evaluate_fid(fdir1, fdir2):
    score = fid.compute_fid(fdir1, fdir2)
    return score


def evaluate_kid(fdir1, fdir2):
    score = fid.compute_kid(fdir1, fdir2) * 1000
    return score


def evaluate_clip(fdir, prompt, model):
    print(".", end="", flush=True)
    cos_sim_list = []
    for img in os.listdir(fdir):
        img = os.path.join(fdir, img)
        cos_sim = model.measure_clip_sim_from_img_and_text(img, prompt)
        cos_sim_list.append(cos_sim)

    clip_score = sum(cos_sim_list) / len(cos_sim_list)
    return clip_score.item() * 100


def eval_experiment(ref_pattern, fake_pattern, output=None):
    clip_evaluator = ClipEvaluator().cuda()
    giqa_evaluator = GIQAEvaluator()
    print_with_box(
        f"Reference: {ref_pattern}\n" f"Fake: {fake_pattern}",
        title="Evaluation Setup",
    )
    # Gather paths
    ref_paths = list(gather_paths(ref_pattern).values())
    fake_dict = gather_paths(fake_pattern)
    num_keys = len(list(fake_dict.keys())[0])

    print_info(f"Number of reference paths: {len(ref_paths)}")
    print_info(f"Number of fake paths: {len(fake_dict)}")
    print_info(f"Number of keys: {num_keys}")

    exp_key_list = []
    for i in range(num_keys - 2):
        exp_key_list.append(natsorted(collect_keys(fake_dict, i)))
    prompts = collect_keys(fake_dict, num_keys - 2)

    result_dict = {}

    for exp_keys in product(*exp_key_list):
        print_info(f"Experiment keys: {exp_keys}")
        exp_keys = list(exp_keys)
        fake_paths = filter_paths(fake_dict, *exp_keys, "*").values()

        # measure fid and kid
        # os.makedirs("temp_ref_dir", exist_ok=True)
        # os.makedirs("temp_fake_dir", exist_ok=True)
        # temp_ref_dir = "temp_ref_dir"
        # temp_fake_dir = "temp_fake_dir"
        # if True:
        with tempfile.TemporaryDirectory() as temp_ref_dir, tempfile.TemporaryDirectory() as temp_fake_dir, tempfile.TemporaryDirectory() as temp_feature_dir:
            for ref_path in ref_paths:
                # concatenate the path into a single filename to prevent collision
                ref_path_str = "_".join(ref_path.split(os.sep))
                os.symlink(
                    ref_path, os.path.join(temp_ref_dir, os.path.basename(ref_path_str))
                )
            for fake_path in fake_paths:
                fake_path_str = "_".join(fake_path.split(os.sep))
                os.symlink(
                    fake_path,
                    os.path.join(temp_fake_dir, os.path.basename(fake_path_str)),
                )

            fid_score = evaluate_fid(temp_fake_dir, temp_ref_dir)
            kid_score = evaluate_kid(temp_fake_dir, temp_ref_dir)
            giqa_scores = giqa_evaluator(temp_ref_dir, temp_fake_dir, temp_feature_dir)
        # print(giqa_scores)
        print_info(f"FID: {fid_score}, KID: {kid_score}, GIQA: {giqa_scores}")

        # measure clip
        clip_scores = []
        for prompt_key in prompts:
            with tempfile.TemporaryDirectory() as temp_fake_dir:
                fake_prompt_paths = filter_paths(
                    fake_dict, *exp_keys, prompt_key
                ).values()
                for fake_path in fake_prompt_paths:
                    fake_path_str = "_".join(fake_path.split(os.sep))
                    os.symlink(
                        fake_path,
                        os.path.join(temp_fake_dir, os.path.basename(fake_path_str)),
                    )
                if not fake_prompt_paths:
                    print_warning(f"No fake images for prompt {prompt_key}")
                    continue

                prompt = prompt_key.replace("_", " ")
                # print_info(f"Prompt: {prompt}")
                clip_score = evaluate_clip(temp_fake_dir, prompt, clip_evaluator)
                # print(f"Prompt: {prompt}, Clip: {clip_score}")
                clip_scores.append(clip_score)
        print_info(f"Number of prompts: {len(clip_scores)}")
        clip_score = sum(clip_scores) / len(clip_scores)
        print_info(f"Average Clip: {clip_score}")

        exp_keys_str = ", ".join(exp_keys)
        result_dict[exp_keys_str] = {
            "fid": fid_score,
            "kid": kid_score,
            "giqa": giqa_scores,
            "clip": clip_score,
        }

    print_with_box(
        f"Results\n"
        + f"{'Experiment':<20}{'FID':<10}{'KID':<10}{'GIQA':<10}{'Clip':<10}\n"
        + f"{'-'*50}\n"
        + "\n".join(
            [
                f"{exp:<20}{scores['fid']:<10.4f}{scores['kid']:<10.4f}{scores['giqa']:<10.4f}{scores['clip']:<10.4f}"
                for exp, scores in result_dict.items()
            ]
        ),
        title="Results",
    )

    if output:
        with open(output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Experiment", "FID", "KID", "GIQA", "Clip"])
            for exp, scores in result_dict.items():
                writer.writerow([exp, scores["fid"], scores["kid"], scores["giqa"], scores["clip"]])
        print_info(f"Results saved in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref_pattern", type=str, required=True)
    parser.add_argument("-f", "--fake_pattern", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    eval_experiment(args.ref_pattern, args.fake_pattern, args.output)
