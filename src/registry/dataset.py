DATASET_REGISTRY = {}

def register_dataset(name):
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper

def get_dataset(name):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered")
    return DATASET_REGISTRY[name]