import torch
import lightning as L
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Callable, Any
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose, Normalize
from src.registry.dataset import register_dataset

from .transforms import (
    NormalizeByChannel,
    StandardizeByHorizon,
    SpectralAugmentation,
    CenteredCropOnLabel,
    ToTensor,
    BinarizeLabel,
    FilterNaNs,
    Standardize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    GaussianNoise,
    RandomCrop,
    CenterCrop,
    SeismicToRGB,
    RandomInvert,
    RandomResize,
    RandomizeScale,
    Resize,
    Unsqueeze,
    Squeeze,
)

from torchio.transforms import RandomElasticDeformation, RandomBlur, RandomGhosting, RandomSpike, OneOf

_augmentations = {
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_vertical_flip": RandomVerticalFlip,
    "random_rotation": RandomRotation,
    "gaussian_noise": GaussianNoise,
    "random_invert": RandomInvert,
    "random_resize": RandomResize,
    "randomize_scale": RandomizeScale,
    "resize": Resize,
    "unsqueeze": Unsqueeze,
    "squeeze": Squeeze,
    "random_elastic_deformation": lambda **kwargs: RandomElasticDeformation(
        include=['data', 'label'], label_keys=['label'], **kwargs
    ),
    "random_blur": lambda **kwargs: RandomBlur(include=['data', 'label'], label_keys=['label'], **kwargs),
    "random_ghosting": lambda **kwargs: RandomGhosting(include=['data', 'label'], label_keys=['label'], **kwargs),
    "random_spike": lambda **kwargs: RandomSpike(include=['data', 'label'], label_keys=['label'], **kwargs),
    "one_of": lambda transforms: OneOf([_augmentations[aug["name"]](**aug.get("params", {})) for aug in transforms]),
}

_default_augmentations = [
    {"name": "random_horizontal_flip", "params": {"prob": 0.5}},
    {"name": "random_rotation", "params": {"degrees": 30}},
    {"name": "gaussian_noise", "params": {"mean": 0.0, "std": 0.5}},
]

_normalization_methods = ['by_channel', 'by_volume']


class SeismicDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        csv_path: str | Path,
        transform: Callable[[dict], Any] | None = None,
        attributes: str | list = 'seismic',
        label_name: str = 'label',
        cache_images: bool = False,
        cache_image_max_size: int = 512,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.attributes = attributes
        self.label_name = label_name
        self.cache_images = cache_images
        self.cache_image_max_size = cache_image_max_size
        self.cache = {} if cache_images else None
        self.cache_resize = (
            Compose([ToTensor(), Resize(size=self.cache_image_max_size)]) if cache_image_max_size is not None else None
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Invalid path for csv file: {self.csv_path}")

        self.fold_paths_df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.fold_paths_df)

    def _get_in_cache(self, idx: int) -> dict[str, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
        else:
            sample = self._get_item_uncached(idx)
            if self.cache_image_max_size is not None and (
                sample["data"].shape[1] > self.cache_image_max_size
                or sample["data"].shape[2] > self.cache_image_max_size
            ):
                sample = self.cache_resize(sample)
                sample["data"] = sample["data"].numpy()
                sample["label"] = sample["label"].numpy()
                self.cache[idx] = sample
            else:
                self.cache[idx] = sample
            return sample

    def _get_item_uncached(self, idx: int) -> dict[str, torch.Tensor]:
        data_path = str(self.root_dir / self.fold_paths_df["relative_path"].iloc[idx])

        if isinstance(self.attributes, str):
            if not os.path.isfile(data_path + f"_{self.attributes}.npy"):
                raise FileNotFoundError(f"Invalid path for sample data: {data_path + f"_{self.attributes}.npy"}")
        else:
            for attribute in self.attributes:
                if not os.path.isfile(data_path + f"_{attribute}.npy"):
                    raise FileNotFoundError(f"Invalid path for sample data: {data_path + f"_{attribute}.npy"}")

        if not os.path.isfile(data_path + f"_{self.label_name}.npy"):
            raise FileNotFoundError(f"Invalid path for sample label: {data_path + f'_{self.label_name}.npy'}")

        if isinstance(self.attributes, str):
            data = np.load(data_path + f"_{self.attributes}.npy")
            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
            elif data.ndim == 3:
                # if data.shape[0] == 3:
                #     data = np.transpose(data, (1, 2, 0))
                pass
            else:
                raise ValueError(f"Invalid data shape for sample data: {data.shape}")

        else:
            data = np.stack(
                [np.load(data_path + f"_{attribute}.npy") for attribute in self.attributes],
                axis=0,
            )

        sample = {
            "data_path": data_path,
            "data": data,
            "label": np.load(data_path + f"_{self.label_name}.npy"),
            "dataset_name": self.fold_paths_df["dataset_name"].iloc[idx],
            "index": idx,
        }

        return sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._get_in_cache(idx) if self.cache_images else self._get_item_uncached(idx)

        if self.transform:
            sample = self.transform(sample)

        return sample


@register_dataset("Seismic")
class SeismicDatasetModule(L.LightningDataModule):
    """
    Torch lightning module for the SeismicDataset class
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.attributes = config["dataset"].get("attributes", 'seismic')

        self.img_size = config["dataset"]["img_size"]
        self.batch_size = config["dataset"]["batch_size"]
        self.num_workers = config["dataset"]["num_workers"]
        self.normalization_method = config["dataset"].get("normalization_method", "by_channel")
        self.means_stds_path = config["dataset"]["means_stds_path"]
        self.cache_images = config["dataset"].get("cache_images", False)
        self.cache_image_max_size = config["dataset"].get("cache_image_max_size", 512)

        self.train_data_dir = Path(config["dataset"]["train"]["data_dir"])
        self.train_fold_paths = Path(config["dataset"]["train"]["csv_path"])

        self.val_data_dir = Path(config["dataset"]["val"]["data_dir"])
        self.val_fold_paths = Path(config["dataset"]["val"]["csv_path"])

        self.test_data_dir = Path(config["dataset"]["test"]["data_dir"])
        self.test_fold_paths = Path(config["dataset"]["test"]["csv_path"])

        self.imagenet_norm = config["dataset"].get("imagenet_norm", False)
        self.centered_crop_on_label = config["dataset"].get("centered_crop_on_label", False)
        self.resize = config["dataset"].get("resize", False)  # can be None, int or tuple

        self.train_attributes = config["dataset"]["train"].get("attributes", self.attributes)
        self.val_attributes = config["dataset"]["val"].get("attributes", self.attributes)
        self.test_attributes = config["dataset"]["test"].get("attributes", self.attributes)

        self.label_name = config["dataset"].get("label_name", "label")
        self.train_label_name = config["dataset"]["train"].get("label_name", self.label_name)
        self.val_label_name = config["dataset"]["val"].get("label_name", self.label_name)
        self.test_label_name = config["dataset"]["test"].get("label_name", self.label_name)

        augmentations_configs = config.get("augmentations", _default_augmentations)

        assert (
            self.normalization_method in _normalization_methods
        ), f"Invalid normalization method: {self.normalization_method}"
        if self.normalization_method == 'by_volume':
            assert (
                "means_stds_path" in config["dataset"]
            ), "means_stds_path must be provided for by_volume normalization"
            normalization_transform = Standardize(self.means_stds_path, std_clip=3)
        elif self.normalization_method == 'by_channel':
            normalization_transform = NormalizeByChannel()

        self.train_transform = [
            ToTensor(),
            BinarizeLabel(threshold=0.1, ignore_index=2),
            FilterNaNs(),
        ]

        if self.resize:
            if isinstance(self.resize, int):
                self.train_transform.append(Resize(size=self.resize))
            elif isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                assert self.resize[0] < self.resize[1], "resize_before_crop min must be less than max"
                self.train_transform.append(RandomResize(min_size=self.resize[0], max_size=self.resize[1]))
            else:
                raise ValueError("resize must be an int or a tuple/list of length 2")

        self.train_transform.append(RandomCrop(size=(self.img_size, self.img_size)))
        self.train_transform.append(normalization_transform)

        augmentations = [
            _augmentations[aug_config["name"]](**aug_config.get("params", {})) for aug_config in augmentations_configs
        ]

        self.train_transform += augmentations

        self.train_transform += [
            SeismicToRGB("min_max" if self.imagenet_norm else "none", data_range=(-3, 3)),
        ]

        self.test_transform = [ToTensor(), BinarizeLabel(threshold=0.1, ignore_index=2), FilterNaNs()]

        if self.resize:
            if isinstance(self.resize, int):
                self.test_transform.append(Resize(size=self.resize))
            elif isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                assert self.resize[0] < self.resize[1], "resize_before_crop min must be less than max"
                self.test_transform.append(Resize(size=self.resize[1]))
            else:
                raise ValueError("resize must be an int or a tuple/list of length 2")

        if self.centered_crop_on_label:
            self.test_transform.append(CenteredCropOnLabel(size=(self.img_size, self.img_size)))
        else:
            self.test_transform.append(CenterCrop(size=(self.img_size, self.img_size)))

        self.test_transform.append(normalization_transform)

        test_augmentations = [
            _augmentations[aug_config["name"]](**aug_config.get("params", {}))
            for aug_config in augmentations_configs
            if aug_config.get("test_apply", False)
        ]
        self.test_transform += test_augmentations

        self.test_transform.append(SeismicToRGB("min_max" if self.imagenet_norm else "none", data_range=(-3, 3)))

        if self.imagenet_norm:
            imagenet_means = [0.485, 0.456, 0.406]
            imagenet_stds = [0.229, 0.224, 0.225]
            self.train_transform.append(Normalize(imagenet_means, imagenet_stds))
            self.test_transform.append(Normalize(imagenet_means, imagenet_stds))

        self.train_transform = Compose(self.train_transform)
        self.test_transform = Compose(self.test_transform)

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = SeismicDataset(
                self.train_data_dir,
                self.train_fold_paths,
                transform=self.train_transform,
                attributes=self.train_attributes,
                label_name=self.train_label_name,
                cache_images=self.cache_images,
                cache_image_max_size=self.cache_image_max_size,
            )
            self.val_dataset = SeismicDataset(
                self.val_data_dir,
                self.val_fold_paths,
                transform=self.test_transform,
                attributes=self.val_attributes,
                label_name=self.val_label_name,
                cache_images=self.cache_images,
                cache_image_max_size=self.cache_image_max_size,
            )
        if stage == "test" or stage is None:
            self.test_dataset = SeismicDataset(
                self.test_data_dir,
                self.test_fold_paths,
                transform=self.test_transform,
                attributes=self.test_attributes,
                label_name=self.test_label_name,
                cache_images=self.cache_images,
                cache_image_max_size=self.cache_image_max_size,
            )

    def train_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True
        )

    def val_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False
        )

    def test_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False
        )