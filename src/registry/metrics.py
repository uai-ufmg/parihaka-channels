import torchmetrics as tm

METRICS_REGISTRY = {
    "precision": tm.Precision,
    "recall": tm.Recall,
    "accuracy": tm.Accuracy,
    "f1": tm.F1Score,
    "iou": tm.JaccardIndex,
    "auroc": tm.AUROC,
}


def register_dataset(name):
    def wrapper(cls):
        METRICS_REGISTRY[name] = cls
        return cls

    return wrapper


def get_dataset(name):
    if name not in METRICS_REGISTRY:
        raise ValueError(f"Metric '{name}' not registered")
    return METRICS_REGISTRY[name]
