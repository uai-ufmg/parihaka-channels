import sys
import os
from argparse import ArgumentParser

from src.config.load import load_config
from src.experiment.runner import run_experiment

# registrar tudo
import src.models
import src.optimizers
import src.criterions
import src.datasets

import yaml

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a deep learning model for seismic segmentation.")
    parser.add_argument("--config", help="Configurarion (.yaml) file to use", type=str, required=True)
    parser.add_argument("--exp_name", help="Experiment name", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dir_path = os.path.join("outputs", args.exp_name)

    run_experiment(config, dir_path)
    print("Training and evaluation completed successfully.")
