from typing import Any, Dict
import inspect

from src.registry.scheduler import get_scheduler


class SchedulerBuilder:
    # Lightning-specific parameters that should NOT be passed to the scheduler
    LIGHTNING_PARAMS = {"type", "interval", "frequency", "monitor", "verbose"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        scheduler_config = self.config.get("solver", {}).get("scheduler", {})
        self.scheduler_name = scheduler_config.get("type", None)

        if self.scheduler_name is None:
            self.scheduler_cls = None
            self.scheduler_params = {}
            return

        self.scheduler_cls = get_scheduler(self.scheduler_name)

        # Filter out Lightning-specific parameters
        all_params = {
            'T_max': self.config['solver']['max_epochs'],
            **{k: v for k, v in scheduler_config.items() if k not in self.LIGHTNING_PARAMS},
        }

        # Further filter to only include parameters accepted by the scheduler
        self.scheduler_params = self._filter_valid_params(all_params)

    def _filter_valid_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only those accepted by the scheduler class."""
        try:
            sig = inspect.signature(self.scheduler_cls.__init__)
            valid_param_names = set(sig.parameters.keys())

            # Keep only parameters that the scheduler accepts
            return {k: v for k, v in params.items() if k in valid_param_names}
        except Exception:
            # If inspection fails, return all params (backward compatibility)
            return params

    def __call__(self, optimizer):
        if self.scheduler_cls is None:
            return None

        return self.scheduler_cls(optimizer, **self.scheduler_params)
