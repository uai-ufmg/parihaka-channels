from src.registry.dataset import DATASET_REGISTRY

class DatasetBuilder:
    def __init__(self, config):
        self.config = config

        if self.config["dataset"]["train"]["name"] not in DATASET_REGISTRY:
            raise ValueError(
                f"Unsupported dataset type: {self.config['dataset']['train']['name']}. Choose from: {list(DATASET_REGISTRY.keys())}."
            )
        self.data_module_cls = DATASET_REGISTRY[self.config["dataset"]["train"]["name"]]

    # ====================
    def __call__(self):
        return self.data_module_cls(self.config)

    