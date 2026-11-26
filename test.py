import glob
import math
import random
import os
import sys
import yaml
import torch
import json
import numpy as np
import torchmetrics as tm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import tqdm
from tabulate import tabulate
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize
from sklearn.metrics import roc_curve, auc

from src.datasets import SeismicDataset
from src.datasets.transforms import (
    ToTensor,
    BinarizeLabel,
    Resize,
    FilterNaNs,
    Standardize,
    SeismicToRGB,
    CenterCrop,
    NormalizeByChannel,
)

from src.utils.plot_confusion_matrix import make_confusion_matrix

# registrar tudo
import src.models
import src.optimizers
import src.criterions
import src.datasets

from src.registry.model import MODEL_REGISTRY
from src.registry.metrics import METRICS_REGISTRY

_data_modules = {
    "seismic": SeismicDataset,
}

_test_modes = ["center", "merge"]

_merge_modes = ["avg", "max"]


def load_model_from_checkpoint(model_name, checkpoint):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_name}. Choose from: {list(MODEL_REGISTRY)}.")
    model = MODEL_REGISTRY[model_name].load_from_checkpoint(checkpoint, weights_only=False)
    return model


def get_metric_args(metric_name, num_classes=1, ignore_index=None, pred_threshold=0.5):
    args = {
        "task": "binary" if num_classes == 1 else "multiclass",
        "num_classes": num_classes,
        "average": "micro",
        "ignore_index": ignore_index,
        "validate_args": False,  # faster to run, harder to debug (change to True if adding a new metric)
    }

    if metric_name not in ["auroc"]:
        args["threshold"] = pred_threshold

    return args


def build_metrics(metrics, num_classes=1, ignore_index=None, prefix="", pred_threshold=0.5):
    return tm.MetricCollection(
        {
            metric: METRICS_REGISTRY[metric](**get_metric_args(metric, num_classes, ignore_index, pred_threshold))
            for metric in metrics
        },
        prefix=prefix,
    )


def build_dataset(
    data_config,
    imagenet_norm=False,
    resize=False,
    center_crop=False,
    attributes="seismic",
    label_attribute="label",
    normalization_method="by_channel",
):
    transforms = [ToTensor()]

    transforms.append(BinarizeLabel(threshold=0.1))
    transforms.append(FilterNaNs())

    if resize:
        transforms.append(Resize(resize))

    if center_crop:
        transforms.append(CenterCrop(size=(center_crop, center_crop)))

    if normalization_method == "by_channel":
        transforms.append(NormalizeByChannel())
    elif normalization_method == "by_volume":
        transforms.append(Standardize(data_config["means_stds_path"], std_clip=3))
    else:
        raise ValueError(
            f"Unsupported normalization method: {normalization_method}. Choose from: ['by_channel', 'by_volume']."
        )

    transforms.append(SeismicToRGB("min_max" if imagenet_norm else "none", data_range=(-3, 3)))

    if imagenet_norm:
        imagenet_means = [0.485, 0.456, 0.406]
        imagenet_stds = [0.229, 0.224, 0.225]
        transforms.append(Normalize(imagenet_means, imagenet_stds))

    transforms = Compose(transforms)

    return _data_modules[data_config["name"]](
        data_config["data_dir"],
        data_config["csv_path"],
        transforms,
        attributes=attributes,
        label_name=label_attribute,
    )


def split_with_overlap(length, region_size=256, min_overlap=64):
    max_stride = region_size - min_overlap
    regions = []
    n_regions = math.ceil(length / region_size)
    if n_regions == 1:
        return [(0, length)]

    stride = (length - region_size) / (n_regions - 1)

    while stride > max_stride:
        n_regions += 1
        stride = (length - region_size) / (n_regions - 1)

    for i in range(n_regions):
        regions.append((round(i * stride), round(i * stride + region_size)))

    return regions


def split_sample(data, label, crop_size=256, overlap=64):
    assert data.shape[1:] == label.shape[1:], "data and label shapes do not match"

    shape = label.shape[1:]

    y_regions = split_with_overlap(shape[0], crop_size, overlap)
    x_regions = split_with_overlap(shape[1], crop_size, overlap)

    regions = []
    for yr in y_regions:
        line = []
        for xr in x_regions:
            line.append((yr, xr))
        regions.extend(line)

    image_batch = []
    label_batch = []
    for r in regions:
        image_batch.append(data[:, r[0][0] : r[0][1], r[1][0] : r[1][1]])
        label_batch.append(label[:, r[0][0] : r[0][1], r[1][0] : r[1][1]])
    image_batch = torch.stack(image_batch)
    label_batch = torch.stack(label_batch)

    return image_batch, label_batch, regions


def merge_prediction(pred, regions, merge_mode="avg"):
    shape = [1, regions[-1][0][-1], regions[-1][1][-1]]

    if merge_mode == "avg":
        full_pred = torch.tensor(np.zeros(shape))
        n_pred = np.zeros(shape)

        for r, p in zip(regions, pred):
            full_pred[:, r[0][0] : r[0][1], r[1][0] : r[1][1]] += p.cpu().detach()
            n_pred[:, r[0][0] : r[0][1], r[1][0] : r[1][1]] += 1

        full_pred = full_pred / n_pred
    elif merge_mode == "max":
        full_pred = torch.full(shape, -float("inf"))

        for r, p in zip(regions, pred):
            full_pred[:, r[0][0] : r[0][1], r[1][0] : r[1][1]] = torch.maximum(
                full_pred[:, r[0][0] : r[0][1], r[1][0] : r[1][1]], p.cpu().detach()
            )
    return full_pred


def plot_batch(
    inputs,
    *labels,
    normalize=True,
    n=4,
    titles=["Input", "Prediction", "Ground Truth"],
    ignore_index=None,
    figsize=None,
    save=None,
    plot_options={"cmap": "viridis"},
    percentiles=(0, 100),
):
    inputs = inputs.cpu().detach().numpy()
    if percentiles is not None:
        pmin, pmax = np.percentile(inputs, percentiles)
        inputs = np.where(inputs < pmin, pmin, inputs)
        inputs = np.where(inputs > pmax, pmax, inputs)
    if normalize:
        in_max = inputs.max()
        in_min = inputs.min()
        inputs = (inputs - in_min) / (in_max - in_min)

    n = min(n, len(inputs), *[len(l) for l in labels])
    plt.figure(figsize=((len(labels) + 1) * 6, n * 3) if figsize is None else figsize)

    for i in range(n):
        for j, data in enumerate([inputs, *labels]):
            plt.subplot(n, 4, i * 4 + j + 1)
            img = data[i]
            if img.shape[0] == 1:  # Grayscale
                img = np.squeeze(img)
                if ignore_index is not None:
                    img = np.ma.masked_equal(img, ignore_index)

                if ((img >= 0) & (img <= 1)).all():
                    plot_options["vmin"] = 0
                    plot_options["vmax"] = 1

                plt.imshow(img, **plot_options)
            else:
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                plt.imshow(img)
            if i == 0:
                plt.title(titles[j])
            plt.axis("off")

    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)


def plot_histogram(
    data,
    bins=32,
    title="Value frequency",
    labels=["Values", "Frequency"],
    figsize=(8, 6),
    save=None,
):
    fig = plt.figure(figsize=figsize)

    plt.hist(data, bins=bins)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)

    if save is None:
        fig.show()
    else:
        fig.savefig(save)


def plot_roc_curve(y_true, y_pred, figsize=(8, 6), save=None, save_thresholds=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)

    fig = plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"ROC curve (auc = {auroc:.02f})", color="darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    if save is None:
        fig.show()
    else:
        fig.savefig(save)

    if save_thresholds is not None:
        thresholds = np.where(np.isinf(thresholds), None, thresholds)

        data = {
            "thresholds": thresholds.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auroc,
        }
        with open(save_thresholds, "w") as f:
            json.dump(data, f, indent=2)


def build_table_from_dict(dict, table_format="github"):
    rows = []
    for name, metrics in dict.items():
        rows.append([name] + [round(metrics[k], 4) for k in metrics])

    headers = ["Dataset", *list(next(iter(dict.values())).keys())]

    return tabulate(rows, headers=headers, tablefmt=table_format)


def print_and_write(string, f):
    print(string)
    f.write(string + "\n")


def test_model(config, model_path, experiment_dir):
    model = load_model_from_checkpoint(config["model"]["name"], model_path)
    model.eval()
    loss = model.loss_fn

    results = {}
    n_classes = config["model"]["num_classes"]
    save_plots = config["test"]["save_plots"]
    n_plots = config["test"]["n_plots"]
    plot_percentiles = config["test"].get("percentiles", (5, 95))
    save_dir = os.path.join(experiment_dir, "test_results")
    plot_roc = config["test"].get("plot_roc", True)
    save_thresholds = config["test"].get("save_thresholds", True)
    default_attributes = config["dataset"].get("attributes", "seismic")
    default_normalization_method = config["dataset"].get("normalization_method", 'by_channel')

    os.makedirs(save_dir, exist_ok=True)

    results_file = os.path.join(save_dir, "results.txt")
    with open(results_file, "w") as f:
        print_and_write(
            f'Testing model {config["model"]["name"]} saved on {model_path} \n',
            f,
        )
        print_and_write(f"Output will be saved on {save_dir} \n\n", f)

    with torch.no_grad():
        for data_config in config["test"]["data"]:
            batch_size = data_config["metric_batch"]
            minibatch_size = data_config["minibatch_size"]
            ignore_idx = data_config["ignore_index"]
            crop_size = data_config["crop_size"]
            crop_overlap = data_config["crop_overlap"]
            plot_seed = data_config["plot_seed"]
            merge_mode = data_config.get("merge_mode", ["avg"])
            test_mode = data_config.get("test_mode", "merge")
            pred_threshold = data_config.get("pred_threshold", 0.5)
            resize = data_config.get('resize', None)
            attributes = data_config.get("attributes", default_attributes)
            label = data_config.get("label", "label")
            normalization_method = data_config.get("normalization_method", default_normalization_method)

            if merge_mode is str:
                merge_mode = [merge_mode]
            assert all([mode in _merge_modes for mode in merge_mode]), f"'merge_mode' must be one of {_merge_modes}"
            assert test_mode in _test_modes, f"'test_mode' must be one of {_test_modes}"
            # script not tested for n_classes > 1

            center_crop = False if test_mode == "merge" else data_config["crop_size"]
            ds = build_dataset(
                data_config,
                config["dataset"].get("imagenet_norm", False),
                resize,
                center_crop,
                attributes=attributes,
                label_attribute=label,
                normalization_method=normalization_method,
            )
            ds_name = data_config["save_name"]

            metrics = [
                build_metrics(
                    config["test"]["metrics"],
                    n_classes,
                    ignore_idx,
                    pred_threshold=pred_threshold,
                )
                for _ in merge_mode
            ]

            has_il = config["test"].get("il_metrics") is not None
            il_threshold = config["test"].get("il_threshold", 0.001)
            il_metrics = [
                build_metrics(
                    config["test"].get("il_metrics", []),
                    n_classes,
                    ignore_idx,
                    "il_",
                    # pred_threshold is il_theshold bc we predict that the image is true iif we predict il_threshold of pixels as channel
                    pred_threshold=il_threshold,
                )
                for _ in merge_mode
            ]

            [m.cuda() for m in metrics]
            [m.reset() for m in metrics]
            [m.cuda() for m in il_metrics]
            [m.reset() for m in il_metrics]

            all_samples = torch.tensor([])
            all_labels = torch.tensor([])
            all_preds = [torch.tensor([]) for _ in merge_mode]
            losses = [torch.tensor([]) for _ in merge_mode]

            channel_amount = []  # percentage of channel in label
            no_channel = []  # 0% of the label is channel
            little_channel = []  # 0-5% of the label is channel
            medium_channel = []  # 5-20% of the label is channel
            large_channel = []  # >20% of the label is channel
            has_channel = []  # >0% of the label is channel
            other = []  # nans

            for i, sample in tqdm(
                enumerate(ds),
                ascii=" >>>=",
                ncols=120,
                desc=f"Testing in dataset {ds_name}",
                total=len(ds),
                file=sys.stdout,
            ):
                if i % batch_size == 0:
                    if i != 0:
                        batch_samples = torch.stack(batch_samples)
                        batch_labels = torch.stack(batch_labels)
                        batch_preds = [torch.stack(preds) for preds in batch_preds]

                        all_samples = torch.cat((all_samples, batch_samples))
                        all_labels = torch.cat((all_labels, batch_labels))
                        batch_labels = batch_labels.cuda()
                        for m, preds in enumerate(batch_preds):
                            all_preds[m] = torch.cat((all_preds[m], preds))
                            preds = preds.cuda()

                            metrics[m].update(preds, batch_labels)
                            losses[m] = torch.cat(
                                (
                                    losses[m],
                                    loss(preds, batch_labels).cpu().view(1),
                                )
                            )

                            if has_il:
                                il_batch_labels = (
                                    (batch_labels == 1).flatten(1).sum(1)
                                    / (batch_labels != ignore_idx).flatten(1).sum(1)
                                    > il_threshold
                                ).float()
                                il_batch_preds = (
                                    (preds.sigmoid() > pred_threshold).float().flatten(1).mean(1) > il_threshold
                                ).float()  # Anteriormente o valor aqui estava contínuo (sem > il_threshold)
                                il_metrics[m].update(il_batch_preds, il_batch_labels)
                    batch_samples = []
                    batch_labels = []
                    batch_preds = [[] for _ in merge_mode]

                x = sample["data"]
                y = sample["label"]

                channel_percentage = (sample["label"] == 1).sum() / (sample["label"] != ignore_idx).sum()
                channel_amount.append(channel_percentage)
                if channel_percentage < il_threshold:
                    no_channel.append(i)
                elif channel_percentage < 0.05:
                    little_channel.append(i)
                elif channel_percentage < 0.1:
                    medium_channel.append(i)
                elif channel_percentage >= 0.1:
                    large_channel.append(i)
                else:
                    other.append(i)

                if channel_percentage > il_threshold:
                    has_channel.append(i)

                batch_samples.append(x)
                batch_labels.append(y)

                X, Y, regions = split_sample(x, y, crop_size=crop_size, overlap=crop_overlap)
                Y_hat = torch.tensor([])

                for mb in range(0, len(regions), minibatch_size):
                    minibatch = X[mb : min(mb + minibatch_size, len(regions))].cuda()
                    Y_hat = torch.cat((Y_hat, model(minibatch).cpu()))

                for m, mode in enumerate(merge_mode):
                    y_hat = merge_prediction(Y_hat, regions, mode)
                    y_hat = torch.where(y == ignore_idx, -9999, y_hat)
                    batch_preds[m].append(y_hat)

            batch_samples = torch.stack(batch_samples)
            batch_labels = torch.stack(batch_labels)
            batch_preds = [torch.stack(preds) for preds in batch_preds]

            all_samples = torch.cat((all_samples, batch_samples))
            all_labels = torch.cat((all_labels, batch_labels))
            batch_labels = batch_labels.cuda()
            for m, preds in enumerate(batch_preds):
                all_preds[m] = torch.cat((all_preds[m], preds))
                preds = preds.cuda()

                metrics[m].update(preds, batch_labels)
                losses[m] = torch.cat(
                    (
                        losses[m],
                        loss(preds, batch_labels).cpu().view(1),
                    )
                )

                if has_il:
                    il_batch_labels = (
                        (batch_labels == 1).flatten(1).sum(1) / (batch_labels != ignore_idx).flatten(1).sum(1)
                        > il_threshold
                    ).float()
                    il_batch_preds = (
                        (preds.sigmoid() > pred_threshold).float().flatten(1).mean(1) > il_threshold
                    ).float()
                    il_metrics[m].update(il_batch_preds, il_batch_labels)

            for i, mode in enumerate(merge_mode):
                ds_full_name = ds_name + "-" + mode if len(merge_mode) > 1 else ds_name
                results[ds_full_name] = {
                    "loss": losses[i].mean().item(),
                    **{m: metrics[i][m].compute().item() for m in metrics[i]},
                    **{m: il_metrics[i][m].compute().item() for m in il_metrics[i]},
                }

                result_str = (
                    f"""Results for dataset `{ds_full_name}`:

- Total images: {len(channel_amount)}
  - Images without channel: {len(no_channel)}
  - Images with channel: {len(has_channel)}
    - Images with small channels (< 5%): {len(little_channel)}
    - Other images medium channels (5-10%): {len(medium_channel)}
    - Other images large channels (> 10%): {len(large_channel)}
  - Other images (ignored labels): {len(other)}

- Loss: \t{losses[i].mean().item():.04f}

- Metrics:\n"""
                    + "\n".join([f"  - {m}: \t{metrics[i][m].compute().item():.04f}" for m in metrics[i]])
                    + "\n"
                    + "\n".join([f"  - {m}: \t{il_metrics[i][m].compute().item():.04f}" for m in il_metrics[i]])
                )

                if save_plots:
                    plt.close("all")
                    plot_dir = os.path.join(save_dir, ds_full_name)
                    os.makedirs(plot_dir, exist_ok=True)

                    channel_percentage = list(
                        filter(
                            lambda x: x > il_threshold,
                            map(lambda x: x * 100, channel_amount),
                        )
                    )
                    plot_histogram(
                        channel_percentage,
                        title="Channel percentage per image",
                        labels=["Channel percentage", "Number of images"],
                        save=os.path.join(plot_dir, "histogram"),
                    )

                    if plot_roc:
                        il_labels = (
                            (all_labels == 1).flatten(1).sum(1) / (all_labels != ignore_idx).flatten(1).sum(1)
                            > il_threshold
                        ).float()
                        il_preds = (all_preds[i].sigmoid() > pred_threshold).flatten(1).float().mean(1)
                        plot_roc_curve(
                            il_labels,
                            il_preds,
                            save=os.path.join(plot_dir, "roc_curve"),
                            save_thresholds=os.path.join(plot_dir, "il_thresholds.json") if save_thresholds else None,
                        )

                    # Salvar plot da matriz de confusão
                    il_labels = (
                        (all_labels == 1).flatten(1).sum(1) / (all_labels != ignore_idx).flatten(1).sum(1)
                        > il_threshold
                    ).int()
                    il_preds = (
                        ((all_preds[i].sigmoid() > pred_threshold).float().flatten(1).mean(1)) > il_threshold
                    ).int()
                    confmat = tm.classification.BinaryConfusionMatrix().to(il_preds.device)

                    il_cm = confmat(il_preds, il_labels).numpy()
                    labels = ['TN', 'FP', 'FN', 'TP']
                    categories = ['0', '1']
                    name_plot_img_cf_matrix = plot_dir + '/il_confusion_matrix.png'
                    make_confusion_matrix(
                        cf=il_cm,
                        group_names=labels,
                        categories=categories,
                        cmap='binary',
                        output_path=name_plot_img_cf_matrix,
                    )

                    groups = [
                        [*no_channel, *has_channel],
                        no_channel,
                        has_channel,
                        little_channel,
                        medium_channel,
                        large_channel,
                    ]
                    group_names = [
                        "all",
                        "no_channel",
                        "has_channel",
                        "little_channel",
                        "medium_channel",
                        "large_channel",
                    ]

                    for samples, group_name in zip(groups, group_names):
                        random.seed(plot_seed)
                        k = min(n_plots, len(samples))
                        if k == 0:
                            continue

                        selected = random.sample(samples, k)
                        plot_batch(
                            all_samples[selected],
                            all_labels[selected],
                            all_preds[i][selected].sigmoid() > pred_threshold,
                            n=k,
                            titles=["Input", "Label", "Prediction"],
                            ignore_index=ignore_idx,
                            save=os.path.join(plot_dir, f"samples_{group_name}"),
                            percentiles=plot_percentiles,
                        )
                with open(results_file, "a") as f:
                    print_and_write(result_str, f)
                    print_and_write("\n======================================== \n", f)

    with open(results_file, "a") as f:
        pretty_table = build_table_from_dict(results)
        tex_table = build_table_from_dict(results, "latex_raw")
        print_and_write("Results Table:", f)
        print_and_write(pretty_table, f)
        f.write("\n\nIn latex format:\n")
        f.write(tex_table)

    with open(os.path.join(save_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def get_model_path(dir_path, model='last'):
    model_paths = glob.glob(os.path.join(dir_path, 'checkpoints', f"{model}*.ckpt"))
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoint found for '{model}' in {os.path.join(dir_path, 'checkpoints')}.")
    model_path = model_paths[-1]
    return model_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Test a deep learning model for seismic segmentation.")
    parser.add_argument("--config", help="Configurarion (.yaml) file to use", type=str, required=True)
    parser.add_argument("--exp_name", help="Experiment name", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dir_path = os.path.join("outputs", args.exp_name)
    model_path = get_model_path(dir_path, model=config["test"].get("model", "last"))

    test_model(config, model_path, dir_path)
    print("Testing completed successfully.")
