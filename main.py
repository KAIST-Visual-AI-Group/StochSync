import os
import argparse
from omegaconf import OmegaConf

from utils.config_utils import load_config
from trainer import Trainer
from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
from k_utils.print_utils import print_with_box, print_info
from k_utils.random_utils import seed_everything

@ignore_kwargs
@dataclass
class Config:
    root_dir: str = "./results/default"
    save_source: bool = True
    seed: int = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    cfg = load_config(args.config, cli_args=extras+["prefix=stage1"])
    
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

    trainer = Trainer(cfg)
    output_filename = trainer.train()

    print_info(f"Training finished. Output saved to {output_filename}")

if __name__ == "__main__":
    main()
