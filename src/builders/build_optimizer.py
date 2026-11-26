from src.registry.optimizer import OPTIMIZER_REGISTRY


class OptimizerBuilder:
    def __init__(self, config):
        self.config = config or {}

        # Strict: require 'solver' key in config
        if "solver" not in self.config:
            raise ValueError("Missing required config key: 'solver'")

        solver_cfg = self.config["solver"]
        name = solver_cfg.get("name")
        if not name or name not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unsupported optimizer type: {name}. Choose from: {list(OPTIMIZER_REGISTRY.keys())}"
            )

        self.optimizer_cls = OPTIMIZER_REGISTRY[name]
        self.solver_cfg = solver_cfg

    def __call__(self, params):
        lr = float(self.solver_cfg.get("learning_rate", 1e-3))
        weight_decay = float(self.solver_cfg.get("weight_decay", 0.0) or 0.0)
        return self.optimizer_cls(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
