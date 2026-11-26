from src.registry.metrics import METRICS_REGISTRY
import torchmetrics as tm


class MetricsBuilder:
    def get_metric_args(self, metric_name):
        args = {
            "task": "binary" if self.config["model"]["num_classes"] == 1 else "multiclass",
            "num_classes": self.config["model"]["num_classes"],
            "average": "micro",
            "ignore_index": self.config["solver"]["ignore_index"],
            "validate_args": False,  # faster to run, harder to debug (change to True if adding a new metric)
        }

        # Only add multidim_average if the metric supports it
        # if metric_name not in ["iou"]:  # replace with the actual ones that fail
        #     args["multidim_average"] = "samplewise"

        return args

    def __init__(self, config):
        self.config = config
        self.metrics = config["logging"].get("metrics", [])
        if type(self.metrics) == str:
            self.metrics = [self.metrics]

        assert type(self.metrics) == list, "'metrics' should be a list"
        for metric in self.metrics:
            assert metric in METRICS_REGISTRY.keys(), f"Unknown metric: {metric}"

    def __call__(self):
        return tm.MetricCollection(
            {metric: METRICS_REGISTRY[metric](**self.get_metric_args(metric)) for metric in self.metrics}
        )
