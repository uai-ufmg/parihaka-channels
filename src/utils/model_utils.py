from prettytable import PrettyTable
import torchvision as tv
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import random


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        table.add_row([name, f'{params}{"" if parameter.requires_grad else "*"}'])
        total_params += params
    print(table)
    print(f"Total Params: {total_params}")
    print(f"Total Trainable Params: {trainable_params}")
    print(f"Percentage of trainable params: {trainable_params*100/total_params:.2f}%")
    return total_params


def plot_predictions(model, images, num_images=16, title="Reconstruction Progress", ignore_index=2, cmap='viridis'):
    n = min(num_images, images['data'].shape[0])  # Limit to num_images images for the grid
    images = {k: v[:n] for k, v in images.items()}  # Take only the first n images
    imgs = images['data'].to(model.device)
    ground_truth = images['label'].cpu().squeeze(1)

    # Ensure model is in eval mode (Lightning handles this usually, but good for safety)
    with torch.no_grad():
        preds = model(imgs).squeeze(1)

    imgs = imgs.cpu()
    preds = preds.cpu()

    # put images in [0,1] range for visualization
    # if model.config["dataset"].get("imagenet_norm", False):
    #     imgs = imgs * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(
    #         1, 3, 1, 1
    #     )
    #     imgs = torch.clamp(imgs, 0, 1)
    # else:
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    preds = (preds > 0).float()
    ground_truth = np.ma.masked_where(ground_truth == ignore_index, ground_truth)

    # Apply the colormap to the ground truth and predictions
    cmap = plt.get_cmap(cmap)
    ground_truth = cmap(ground_truth)[..., :3]
    ground_truth = torch.from_numpy(ground_truth).permute(0, 3, 1, 2).float()
    preds = cmap(preds)[..., :3]
    preds = torch.from_numpy(preds).permute(0, 3, 1, 2).float()

    comparison = torch.cat([imgs, ground_truth, preds])

    # make_grid arranges them. nrow=16 means the top row is all originals, midle row is all ground truth, and the bottom row is all reconstructions
    grid = tv.utils.make_grid(comparison, nrow=n, normalize=True)

    # Access the tensorboard logger specifically
    if model.logger is not None:
        tensorboard = model.logger.experiment
        tensorboard.add_image(title, grid, global_step=model.global_step)


def select_images_for_logging(dl, num_positive=8, num_negative=8, threshold=0.1, ignore_index=2, seed=16):
    positive_images = []
    negative_images = []

    for batch in dl:
        data, labels = batch['data'], batch['label']

        for img, label in zip(data, labels):
            positive_ratio = (label == 1).sum().float() / (label != ignore_index).sum().float()

            if positive_ratio > threshold:
                positive_images.append({'data': img, 'label': label})
            elif positive_ratio <= threshold:
                negative_images.append({'data': img, 'label': label})

    random.seed(seed)
    random.shuffle(positive_images)
    random.shuffle(negative_images)
    positive_images = positive_images[:num_positive]
    negative_images = negative_images[:num_negative]

    return {
        'data': torch.stack([img['data'] for img in positive_images + negative_images]),
        'label': torch.stack([img['label'] for img in positive_images + negative_images]),
    }
