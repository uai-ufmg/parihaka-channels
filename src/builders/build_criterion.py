from src.registry.criterion import CRITERION_REGISTRY

class CriterionBuilder:
    def __init__(self, config):
        self.config = config

        if self.config["solver"]["loss"] not in CRITERION_REGISTRY:
            raise ValueError(f"Unsupported loss type: {self.config['solver']['loss']}. Choose from: {list(CRITERION_REGISTRY.keys())}.")
        else:
            self.criterion_cls = CRITERION_REGISTRY[self.config["solver"]["loss"]]

    # ====================
    def __call__(self):
        return self.criterion_cls(
                mode="binary" if self.config["model"]["num_classes"] == 1 else "multiclass",
                ignore_index=self.config["solver"]["ignore_index"],
                from_logits=True,
            )

    