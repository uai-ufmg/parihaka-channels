import segmentation_models_pytorch as smp

CRITERION_REGISTRY = {
    "dice": smp.losses.DiceLoss,
}


def register_criterion(name):
    def wrapper(cls):
        CRITERION_REGISTRY[name] = cls
        return cls

    return wrapper


def get_criterion(name):
    if name not in CRITERION_REGISTRY:
        raise ValueError(f"Criterion '{name}' not registered")
    return CRITERION_REGISTRY[name]
