from src.registry.model import MODEL_REGISTRY

class ModelBuilder:
    def __init__(self, config):
        self.config = config
        if self.config["model"]["name"] not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {self.config["model"]['name']}. Choose from: {list(MODEL_REGISTRY.keys())}.")
        if self.config["model"].get("checkpoint") is None:
            self.model_cls = MODEL_REGISTRY[self.config["model"]["name"]]
        else:
            model = self.load_model_from_checkpoint(self.config["model"]["name"], self.config["model"].get("checkpoint"))

    def __call__(self, loss, optim_builder, metrics, scheduler_builder=None, scheduler_config=None):
        self.config["model"]["logging"] = self.config.get("logging", {})
        return self.model_cls(self.config["model"], loss, optim_builder, metrics, scheduler_builder, scheduler_config)

    def load_model_from_checkpoint(self, model_name, checkpoint):
        model = MODEL_REGISTRY[model_name].load_from_checkpoint(checkpoint)
        return model

    