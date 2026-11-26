import torch
import lightning as L
import yaml
from argparse import ArgumentParser
from typing import Any, Tuple
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from src.callbacks.image_logging import ImageLoggingCallback

from src.registry.dataset import DATASET_REGISTRY
from src.registry.model import MODEL_REGISTRY
from src.registry.criterion import CRITERION_REGISTRY
from src.registry.optimizer import OPTIMIZER_REGISTRY
from src.registry.scheduler import SCHEDULER_REGISTRY

from src.builders import (
    OptimizerBuilder,
    SchedulerBuilder,
    ModelBuilder,
    CriterionBuilder,
    DatasetBuilder,
    MetricsBuilder,
)
import os

import random
import numpy as np


def set_seed(seed: int = 42):
    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    try:
        import torch

        # PyTorch (CPU)
        torch.manual_seed(seed)

        # PyTorch (GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Para maior determinismo (pode reduzir performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        pass

    print(f"Seed fixada em {seed}")


def build_experiment(config: dict[str, Any], output_dir: str) -> Tuple:
    """
    Build the complete experiment pipeline from configuration.

    Args:
        config: Configuration dictionary loaded from YAML
        output_dir: Directory for outputs and checkpoints

    Returns:
        Tuple of (model, data_module, trainer, callbacks)
    """

    set_seed(config["misc"]["seed"])
    # ===========================================
    # Data Module building
    # ===========================================
    data_module_builder = DatasetBuilder(config)
    data_module = data_module_builder()

    # ===========================================
    # Criterion building
    # ===========================================
    criterion_builder = CriterionBuilder(config)
    criterion = criterion_builder()

    # ===========================================
    # Optimizer Builder
    # ===========================================
    optim_builder = OptimizerBuilder(config)

    # ===========================================
    # Scheduler Builder (NEW!)
    # ===========================================
    scheduler_builder = SchedulerBuilder(config)
    scheduler_config = config.get("solver", {}).get("scheduler", {})

    # ===========================================
    # Metrics building
    # ===========================================
    metrics = MetricsBuilder(config)
    metrics = metrics()

    # ===========================================
    # Model building
    # ===========================================
    model_builder = ModelBuilder(config)
    model = model_builder(
        criterion, optim_builder, metrics, scheduler_builder=scheduler_builder, scheduler_config=scheduler_config
    )

    # ===========================================
    # Trainer Setup
    # ===========================================
    logger = TensorBoardLogger(output_dir, name="tensorboard")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="best-epoch-{epoch:02d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        save_last=True,
    )

    logging_cfg = config.get("logging", {})
    image_logging_callback = ImageLoggingCallback(
        num_images=logging_cfg.get("num_images", 16),
        num_positive=logging_cfg.get("num_positive", 8),
        num_negative=logging_cfg.get("num_negative", 8),
        threshold=logging_cfg.get("threshold", 0.1),
        ignore_index=logging_cfg.get("ignore_index", 0),
        cmap=logging_cfg.get("cmap", "viridis"),
    )

    trainer = L.Trainer(
        max_epochs=config['solver']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, image_logging_callback],
        logger=logger,
        log_every_n_steps=logging_cfg.get("log_every_n_steps", 50),
    )

    return model, data_module, trainer, [checkpoint_callback]


def run_experiment(config: dict[str, Any], dir_path: str) -> None:
    """Run a complete training and evaluation experiment."""
    config_path = os.path.join(dir_path, "config.yaml")
    os.makedirs(dir_path, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    model, data_module, trainer, callbacks = build_experiment(config, output_dir=dir_path)

    trainer.fit(model, datamodule=data_module, ckpt_path=config["model"].get("checkpoint"))
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a deep learning model for seismic segmentation.")
    parser.add_argument("--config", help="Configuration (.yaml) file to use", type=str, required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # default output dir behavior kept for compatibility
    exec_id, config_name = args.config.split("/")[-2:]
    config_number = config_name.split("configs_")[1].split(".")[0]
    dir_path = "outputs/" + exec_id + "/results/" + config_number

    run_experiment(config, dir_path=dir_path)
    print("Training and evaluation completed successfully.")
