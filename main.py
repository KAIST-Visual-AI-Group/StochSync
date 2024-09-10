import os
import argparse
from omegaconf import OmegaConf
from datetime import datetime
from time import time

import torch

from utils.config_utils import load_config
from distillation_trainer import DistillationTrainer
from general_trainer import GeneralTrainer
from rewind_trainer import RewindTrainer
from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
from k_utils.print_utils import print_with_box, print_info
from k_utils.random_utils import seed_everything


@ignore_kwargs
@dataclass
class Config:
    root_dir: str = "./results/default"
    save_source: bool = False
    seed: int = 0
    tag: str = ""


def main():
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument(
        "-t",
        "--trainer_type",
        default="general",
        choices=["general", "distillation", "rewind"],
        help="type of trainer to use",
    )
    args, extras = parser.parse_known_args()
    cfg = load_config(args.config, cli_args=extras)
    
    now = datetime.now()
    strnow = now.strftime("%Y%m%d_%H%M%S")
    
    cfg.root_dir = os.path.join(cfg.root_dir.replace(" ", "_"), cfg.tag, strnow)

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

    if args.trainer_type == "distillation":
        trainer = DistillationTrainer(cfg)
    elif args.trainer_type == "general":
        trainer = GeneralTrainer(cfg)
    elif args.trainer_type == "rewind":
        trainer = RewindTrainer(cfg)
    else:
        raise ValueError(f"Unknown trainer type: {args.trainer}")

    # seed_everything(main_cfg.seed)
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


if __name__ == "__main__":
    main()
