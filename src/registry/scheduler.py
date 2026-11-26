"""Scheduler registry for learning rate schedulers.

This module provides a registry system for learning rate schedulers,
allowing flexible configuration and extensibility.

Usage:
    scheduler_class = get_scheduler("cosine")
    
    @register_scheduler("custom_scheduler")
    class MyCustomScheduler:
        def __init__(self, optimizer, **kwargs):
            ...
"""

from typing import Callable, Dict, Any

SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_scheduler(name: str):
    """Decorator to register a scheduler class.
    
    Args:
        name: Name to register the scheduler under.
    """
    def wrapper(cls):
        SCHEDULER_REGISTRY[name] = cls
        return cls
    return wrapper


def get_scheduler(name: str) -> Callable:
    """Get a scheduler class from the registry.
    
    Args:
        name: Name of the scheduler to retrieve.
    
    Returns:
        Scheduler class.
    
    Raises:
        ValueError: If scheduler is not registered.
    """
    if not SCHEDULER_REGISTRY:
        try:
            import src.schedulers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Could not import src.schedulers to populate the scheduler registry. "
                "Check that the src/schedulers package exists and is importable."
            ) from exc

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Scheduler '{name}' not registered. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[name]


