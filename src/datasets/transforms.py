import torch
import random
import numpy as np
import json
import os

try:
    from kornia.enhance import equalize_clahe
except ImportError:
    print("Retrying Kornia equalize_clahe import.")
    # for some reason kornia fails in the first import attempt, but works in the second one
    try:
        from kornia.enhance import equalize_clahe
    except ImportError:
        print("Kornia equalize_clahe import error. You won't be able to use the clahe")

from einops import rearrange
from pathlib import Path
from torchvision.transforms import v2
import torch.nn.functional as F
import torchvision.transforms.functional as tF
from torchvision.transforms import InterpolationMode

_interpolations = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'nearest_exact': InterpolationMode.NEAREST_EXACT,
}


class NormalizeByChannel:
    def __init__(self, mode="mean_std", clip_percentiles=None, eps=1e-6):  # ex: (1, 99)
        assert mode in ["mean_std", "median_iqr", "minmax"]
        self.mode = mode
        self.clip_percentiles = clip_percentiles
        self.eps = eps

    def __call__(self, sample):
        x = sample["data"]  # (C, H, W)

        # -------------------------
        # Percentile clipping (por canal)
        # -------------------------
        if self.clip_percentiles is not None:
            p_low, p_high = self.clip_percentiles

            x_flat = x.view(x.shape[0], -1)

            low = x_flat.quantile(p_low / 100.0, dim=1).view(-1, 1, 1)
            high = x_flat.quantile(p_high / 100.0, dim=1).view(-1, 1, 1)

            x = torch.clamp(x, min=low, max=high)

        # -------------------------
        # Normalização
        # -------------------------
        if self.mode == "mean_std":
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True)
            x = (x - mean) / (std + self.eps)
            x = x / (x.abs().mean(dim=(1, 2), keepdim=True) + self.eps)

        elif self.mode == "median_iqr":
            x_flat = x.view(x.shape[0], -1)

            median = x_flat.median(dim=1).values.view(-1, 1, 1)

            q1 = x_flat.quantile(0.25, dim=1).view(-1, 1, 1)
            q3 = x_flat.quantile(0.75, dim=1).view(-1, 1, 1)

            iqr = q3 - q1

            x = (x - median) / (iqr + self.eps)

        elif self.mode == "minmax":
            x_flat = x.view(x.shape[0], -1)

            min_val = x_flat.min(dim=1).values.view(-1, 1, 1)
            max_val = x_flat.max(dim=1).values.view(-1, 1, 1)

            x = (x - min_val) / (max_val - min_val + self.eps)

        sample["data"] = x
        return sample


class CenteredCropOnLabel:
    def __init__(self, crop_size):
        self.crop_h, self.crop_w = crop_size

    def __call__(self, sample):
        image = sample["data"]  # (C, H, W)
        mask = sample["label"][0]  # (H, W)

        C, H, W = image.shape
        ys, xs = torch.where(mask == 1)

        if ys.numel() == 0:
            center_y = H // 2
            center_x = W // 2
        else:
            center_y = ys.float().mean().round().long()
            center_x = xs.float().mean().round().long()

        center_y = int(center_y)
        center_x = int(center_x)

        y1 = max(0, min(center_y - self.crop_h // 2, H - self.crop_h))
        x1 = max(0, min(center_x - self.crop_w // 2, W - self.crop_w))

        y2 = y1 + self.crop_h
        x2 = x1 + self.crop_w

        # Crop garantido
        cropped_image = image[:, y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        assert cropped_image.shape[1:] == (self.crop_h, self.crop_w), f"Crop errado: {cropped_image.shape}"

        sample["data"] = cropped_image
        sample["label"] = cropped_mask.unsqueeze(0)

        return sample


class SpectralAugmentation:
    def __init__(
        self,
        p_drop=0.3,
        p_perturb=0.9,
        p_invert=0.2,
        scale_range=(0.9, 1.1),
        shift_range=(-0.1, 0.1),
    ):
        self.p_drop = p_drop
        self.p_perturb = p_perturb
        self.p_invert = p_invert
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, sample):
        # img: [3, H, W]
        img = sample["data"].clone()

        # --- Ideia 1: Dropout de canal ---
        if random.random() < self.p_drop:
            c = random.randint(0, img.shape[0] - 1)
            img[c] = 0.0

        # --- Ideia 2: Perturbação por canal ---
        if random.random() < self.p_perturb:
            for c in range(img.shape[0]):
                scale = random.uniform(*self.scale_range)
                shift = random.uniform(*self.shift_range)
                img[c] = img[c] * scale + shift

        # --- Nova ideia: inversão de sinal (global) ---
        if random.random() < self.p_invert:
            img = -img
        sample["data"] = img

        return sample


class ToTensor:
    """
    Converts numpy arrays in a sample into torch tensors
    """

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        sample = sample.copy()

        if sample["data"].ndim == 2:
            sample["data"] = rearrange(sample["data"], "h w -> 1 h w")
        elif sample["data"].ndim == 3:
            pass
        else:
            raise ValueError(f"Invalid data shape for sample data: {sample['data'].shape}")
        if sample["label"].ndim == 2:
            sample["label"] = rearrange(sample["label"], "h w -> 1 h w")
        elif sample["label"].ndim == 3:
            if sample["label"].shape[0] != 1:
                raise ValueError(f"Invalid label shape for sample label: {sample['label'].shape}")
        else:
            raise ValueError(f"Invalid data shape for sample label: {sample['label'].shape}")

        sample["data"] = torch.tensor(sample["data"].copy().astype(np.float32))
        sample["label"] = torch.tensor(sample["label"].copy().astype(np.float32))

        return sample


class BinarizeLabel:
    """
    Turns the label from continous to binary values according to a threshold
    """

    def __init__(self, threshold: float = 0.1, ignore_index: int = None):
        self.threshold = threshold
        self.ignore_index = ignore_index

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.threshold > 0 and self.threshold < 1

        label = sample["label"]
        ignore_mask = torch.zeros_like(label, dtype=torch.bool)
        if self.ignore_index is not None:
            ignore_mask = label == self.ignore_index
        label = torch.where((label >= self.threshold) & ~ignore_mask, 1, label)
        label = torch.where((label < self.threshold) & ~ignore_mask, 0, label)
        sample["label"] = label.float()

        return sample


class FilterNaNs:
    """
    Turns NaNs from data to 0 and label to 2.
    """

    def __init__(self, data_nan=0.0, label_nan=2.0):
        self.data_nan = data_nan
        self.label_nan = label_nan

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["data"] = torch.nan_to_num(sample["data"], nan=self.data_nan)
        sample["label"] = torch.nan_to_num(sample["label"], nan=self.label_nan)

        return sample


class Standardize:
    """
    Performs a z-normalization on data based on precomputed values.
    """

    def __init__(
        self, means_stds_path: str | Path, std_clip: int | None = None, attributes: list[str] = ['seismic']
    ) -> None:
        self.std_clip = std_clip
        self.attributes = attributes

        if not os.path.isfile(means_stds_path):
            raise FileNotFoundError(f"Invalid path for means_stds json file: {means_stds_path}")

        with open(means_stds_path, "r") as f:
            self.means_stds = json.load(f)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dataset_name = sample["dataset_name"]
        for i, attribute in enumerate(self.attributes):
            mean = self.means_stds[dataset_name][attribute]["mean"]
            mean = rearrange(np.array([mean]).astype(np.float32), "c -> c 1 1")

            std = self.means_stds[dataset_name][attribute]["std"]
            std = rearrange(np.array([std]).astype(np.float32), "c -> c 1 1")

            sample["data"][i] = (sample["data"][i] - mean) / std

        # Clips values outside of specified number of standard deviations
        if self.std_clip is not None:
            sample["data"] = torch.clamp(sample["data"], min=-self.std_clip, max=self.std_clip)

        return sample


class StandardizeByHorizon:
    """
    Performs a z-normalization on data based on precomputed values.
    """

    def __init__(self, std_clip: int | None = None, attributes: list[str] = ['seismic']) -> None:
        self.std_clip = std_clip
        self.attributes = attributes

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dataset_name = sample["dataset_name"]
        for i, attribute in enumerate(self.attributes):
            mean = sample["data"][i].mean().item()
            mean = rearrange(np.array([mean]).astype(np.float32), "c -> c 1 1")

            std = sample["data"][i].std().item()
            std = rearrange(np.array([std]).astype(np.float32), "c -> c 1 1")

            sample["data"][i] = (sample["data"][i] - mean) / std

        # Clips values outside of specified number of standard deviations
        if self.std_clip is not None:
            sample["data"] = torch.clamp(sample["data"], min=-self.std_clip, max=self.std_clip)

        return sample


class RandomHorizontalFlip:
    """
    Performs a horizontal flip of data and label with a given probability
    """

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if np.random.rand() < self.prob:
            sample["data"] = v2.functional.hflip(sample["data"])
            sample["label"] = v2.functional.hflip(sample["label"])

        return sample


class RandomVerticalFlip:
    """
    Performs a vertical flip of data and label with a given probability
    """

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if np.random.rand() < self.prob:
            sample["data"] = v2.functional.vflip(sample["data"])
            sample["label"] = v2.functional.vflip(sample["label"])

        return sample


class RandomRotation:
    """
    Performs a random rotation of data and label according to a specified degree
    """

    def __init__(self, degrees: int = 30) -> None:
        self.degrees = degrees

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        angle = v2.RandomRotation.get_params([-self.degrees, self.degrees])

        sample["data"] = v2.functional.rotate(sample["data"], angle)
        sample["label"] = v2.functional.rotate(sample["label"], angle)

        return sample


class GaussianNoise:
    """
    Applies gaussian noise to samples
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["data"] += torch.normal(self.mean, self.std, size=sample["data"].shape)

        return sample


class RandomCrop:
    """
    Crops images at random location to the desired new shape
    """

    def __init__(self, size: tuple[int, int]) -> None:
        assert len(size) == 2

        self.new_height, self.new_width = size

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data, label = sample["data"], sample["label"]

        assert len(data.shape) == len(label.shape) == 3
        _, h, w = data.shape

        assert (
            self.new_height <= h
        ), f"New height {self.new_height} is larger than original height {h}, input sample shape: {data.shape}"
        assert (
            self.new_width <= w
        ), f"New width {self.new_width} is larger than original width {w}, input sample shape: {data.shape}"

        top = int(np.random.randint(0, h - self.new_height + 1))
        left = int(np.random.randint(0, w - self.new_width + 1))

        sample["data"] = data[:, top : top + self.new_height, left : left + self.new_width]
        sample["label"] = label[:, top : top + self.new_height, left : left + self.new_width]

        return sample


class CenterCrop:
    """
    Crops images at the center to the desired new shape
    """

    def __init__(self, size: tuple[int, int]) -> None:
        assert len(size) == 2

        self.new_height, self.new_width = size

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data, label = sample["data"], sample["label"]

        assert len(data.shape) == len(label.shape) == 3
        _, h, w = data.shape

        assert (
            self.new_height <= h
        ), f"New height {self.new_height} is larger than original height {h}, input sample shape: {data.shape}"
        assert (
            self.new_width <= w
        ), f"New width {self.new_width} is larger than original width {w}, input sample shape: {data.shape}"

        top = (h - self.new_height) // 2
        left = (w - self.new_width) // 2

        sample["data"] = data[:, top : top + self.new_height, left : left + self.new_width]
        sample["label"] = label[:, top : top + self.new_height, left : left + self.new_width]

        return sample


class SeismicToRGB:
    """
    Converts single channel seismic images to RGB
    """

    def __init__(self, norm: 'str' = 'none', data_range=None) -> None:
        assert norm in ['none', 'min_max', 'jpeg'], 'norm must be one of ["none", "min_max", "jpeg"]'

        self.norm = norm
        self.range = data_range

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data = sample["data"].squeeze()

        if self.norm == 'min_max' or self.norm == 'jpeg':
            # Normalizing data to the [0, 1] range
            if self.range is None:
                min_value = torch.min(data)
                max_value = torch.max(data)
            else:
                min_value = self.range[0]
                max_value = self.range[1]
            data = (data - min_value) / (max_value - min_value + 1e-8)

        if self.norm == 'jpeg':
            # Moving the range to [0, 255]
            data = (data * 255).to(torch.uint8)

        # Replicating the single color channel
        if data.ndim == 2:
            sample["data"] = torch.stack([data] * 3, axis=0).to(torch.float32)
        else:
            sample["data"] = data.to(torch.float32)

        return sample


class RandomResize:
    def __init__(self, min_size, max_size, interpolation='bilinear', label_interpolation='nearest'):
        self.min_size = min_size
        self.max_size = max_size
        assert (
            interpolation in _interpolations
        ), f"Invalid interpolation method: {interpolation}. Supported methods: {list(_interpolations.keys())}"
        self.interpolation = _interpolations[interpolation]
        assert (
            label_interpolation in _interpolations
        ), f"Invalid label interpolation method: {label_interpolation}. Supported methods: {list(_interpolations.keys())}"
        self.label_interpolation = _interpolations[label_interpolation]

    def __call__(self, sample: dict[str, torch.Tensor]):
        target_size = int(torch.randint(self.min_size, self.max_size, ()))

        sample['data'] = tF.resize(sample['data'], size=target_size, interpolation=self.interpolation)
        sample['label'] = tF.resize(sample['label'], size=target_size, interpolation=self.label_interpolation)

        return sample


class Resize:
    def __init__(self, size, interpolation='bilinear', label_interpolation='nearest'):
        self.size = size
        assert (
            interpolation in _interpolations
        ), f"Invalid interpolation method: {interpolation}. Supported methods: {list(_interpolations.keys())}"
        self.interpolation = _interpolations[interpolation]
        assert (
            label_interpolation in _interpolations
        ), f"Invalid label interpolation method: {label_interpolation}. Supported methods: {list(_interpolations.keys())}"
        self.label_interpolation = _interpolations[label_interpolation]

    def __call__(self, sample: dict[str, torch.Tensor]):
        sample['data'] = tF.resize(sample['data'], size=self.size, interpolation=self.interpolation)
        sample['label'] = tF.resize(sample['label'], size=self.size, interpolation=self.label_interpolation)

        return sample


class RandomInvert:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: dict[str, torch.Tensor]):
        flip = bool(torch.rand(()) > self.p)

        if flip:
            sample['data'] = -sample['data']

        return sample


class Unsqueeze:
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, sample: dict[str, torch.Tensor]):
        sample['data'] = sample['data'].unsqueeze(self.dim)
        sample['label'] = sample['label'].unsqueeze(self.dim)

        return sample


class Squeeze:
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, sample: dict[str, torch.Tensor]):
        sample['data'] = sample['data'].squeeze(self.dim)
        sample['label'] = sample['label'].squeeze(self.dim)

        return sample


### NOVAS TRANSFORMAÇÔES


class RandomizeScale:
    """
    Augmentation that changes the scale of the sample to make the model scale free.
    """

    def __init__(
        self, scale_factor: float | None = None, max_offset: float | None = None, randomize_signal: bool | None = False
    ) -> None:
        self.scale_factor = scale_factor
        self.max_offset = max_offset
        self.randomize_signal = randomize_signal

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # scale = torch.random.uniform(1/self.factor, self.factor)
        scale = torch.empty(1).uniform_(1 / self.scale_factor, self.scale_factor).item()
        sample["data"] = sample["data"] * scale

        offset = torch.empty(1).uniform_(-self.max_offset, self.max_offset).item()
        sample["data"] = sample["data"] + offset

        if self.randomize_signal:
            signal = np.random.choice([-1, 1])
            sample["data"] = sample["data"] * signal

        return sample


class ScaleResolution:
    """
    Deterministic transformation that changes image resolution
    by scaling with a fixed factor, without restoring to original size.
    """

    def __init__(self, means_stds_path: str | Path) -> None:

        if not os.path.isfile(means_stds_path):
            raise FileNotFoundError(f"Invalid path for means_stds json file: {means_stds_path}")

        with open(means_stds_path, "r") as f:
            self.means_stds_dict = json.load(f)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        factor = self.means_stds_dict[sample["dataset_name"]]['resolution']

        img = sample["data"]  # expects [C, H, W]
        _, H, W = img.shape

        new_H, new_W = max(1, int(H * factor)), max(1, int(W * factor))

        # --- Image ---
        img_scaled = F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False)[0]
        sample["data"] = img_scaled

        # --- Label ---
        label = sample["label"]  # expects [H, W] or [1, H, W]
        if label.ndim == 2:
            label = label.unsqueeze(0)  # [1, H, W]
        label_scaled = F.interpolate(label.unsqueeze(0).float(), size=(new_H, new_W), mode="nearest")[0].long()

        if sample["label"].ndim == 2:
            label_scaled = label_scaled[0]

        sample["label"] = label_scaled

        return sample


class EnrichChannels:
    """
    Multiplica sample["data"][0] por -1 e aplica a exponencial.
    Assume que sample["data"] é um numpy array ou tensor.
    """

    def __init__(self, channel_enrichment=True):
        self.channel_enrichment = channel_enrichment

    def __call__(self, sample: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.channel_enrichment == False:
            return sample

        # Exponencial enrichment
        data = sample["data"][0]
        sample["data"][1] = torch.exp(-data)
        sample["data"][2] = torch.exp(data)

        sample["data"][1] = sample["data"][1] - sample["data"][1].mean()
        sample["data"][1] = sample["data"][1] / (sample["data"][1].std() + 1e-8)
        sample["data"][1] = sample["data"][1].clip(-3, 3)

        sample["data"][2] = sample["data"][2] - sample["data"][2].mean()
        sample["data"][2] = sample["data"][2] / (sample["data"][2].std() + 1e-8)
        sample["data"][2] = sample["data"][2].clip(-3, 3)

        # Exponencial enrichment

        return sample


class CLAHE:

    def __init__(
        self,
        clip_limit: int | float = 2,
        grid_size: tuple[int, int] | list[int] = (8, 8),
        data_range: None | tuple[int, int] | list[int] = None,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, list):
            grid_size = tuple(grid_size)
            assert len(grid_size) == 2, "grid_size must be a tuple of (height, width)"
        self.clip_limit, self.grid_size = float(clip_limit), grid_size
        if data_range is not None:
            assert (
                isinstance(data_range, (tuple, list)) and len(data_range) == 2
            ), "data_range must be a tuple or list of (min, max)"
            data_range = tuple(data_range)
        else:
            data_range = (0, 1)

        self.data_range = data_range

    def __call__(self, sample: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["data"] = (sample["data"] - self.data_range[0]) / (self.data_range[1] - self.data_range[0] + 1e-8)
        sample["data"] = sample["data"].clip(0, 1)
        sample["data"] = equalize_clahe(sample["data"], clip_limit=self.clip_limit, grid_size=self.grid_size)
        sample["data"] = sample["data"] * (self.data_range[1] - self.data_range[0]) + self.data_range[0]

        return sample

    def __repr__(self) -> str:
        return "{}(clip_limit={}, tile_grid_size={})".format(self.__class__.__name__, self.clip_limit, self.grid_size)
