import os
import argparse
from time import time
from datetime import datetime
import glob 
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from stochsync.utils.config_utils import load_config
from stochsync.general_trainer import GeneralTrainer
from stochsync.utils.extra_utils import ignore_kwargs
from stochsync.utils.print_utils import print_with_box, print_info
from stochsync.utils.random_utils import seed_everything


@ignore_kwargs
@dataclass
class Config:
    root_dir: str = "./results/default"
    save_source: bool = False
    seed: int = 1
    tag: str = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument(
        "-t",
        "--trainer_type",
        default="general",
        choices=["general", "distillation"],
        help="type of trainer to use",
    )
    args, extras = parser.parse_known_args()
    cfg = load_config(args.config, cli_args=extras)
    
    now = datetime.now()
    strnow = now.strftime("%Y%m%d_%H%M%S")
    
    cfg.root_dir = os.path.join(cfg.root_dir.replace(" ", "_"), cfg.tag)
    
    if os.path.exists(os.path.join(cfg.root_dir, "_output")):
        cfg.root_dir = cfg.root_dir + "_" + strnow

    print_with_box(
        f"Config loaded from {args.config} with the following content:\n{OmegaConf.to_yaml(cfg)}",
        title="Config",
        max_len=88,
    )
    main_cfg = Config(**cfg)
    seed_everything(main_cfg.seed)

    # save the config to a file
    os.makedirs(main_cfg.root_dir, exist_ok=True)
    with open(os.path.join(main_cfg.root_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print_info("trainer_type", args.trainer_type)
    if args.trainer_type == "general":
        trainer = GeneralTrainer(cfg)
    elif args.trainer_type == "distillation":
        # trainer = DistillationTrainer(cfg)
        raise ValueError("DistillationTrainer is deprecated. Please use GeneralTrainer instead.")
    else:
        raise ValueError(f"Unknown trainer type: {args.trainer}")

    start_time = time()
    output_filename = trainer.train()
    collapse_time = time() - start_time
    print_with_box(
        (
            f"Training finished in {collapse_time:.2f} seconds.\n"
            f"Output saved to {output_filename}"
        ),
        title="DistillAnything Results",
    )
    # save the time taken to train under the root directory
    with open(os.path.join(main_cfg.root_dir, "time.txt"), "w") as f:
        f.write(f"{collapse_time:.3f}\n")


if __name__ == "__main__":
    main()
