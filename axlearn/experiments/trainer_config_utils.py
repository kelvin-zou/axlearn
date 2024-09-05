# Copyright © 2023 Apple Inc.

"""Trainer config utilities."""
from functools import wraps
from typing import Callable, List, Optional

from typing_extensions import Protocol

from axlearn.common.config import InstantiableConfig, ModuleOverride


class TrainerConfigFn(Protocol):
    """A TrainerConfigFn takes a data_dir as argument and returns a Config for instantiating a
    Trainer, e.g. SpmdTrainer.
    """

    # Note: avoid using SpmdTrainer.Config so we don't need to introduce a dependency to trainer.
    # This also makes it possible to define custom trainers with the same protocol.
    def __call__(self, data_dir: Optional[str] = None) -> InstantiableConfig:
        ...


def with_overrides(trainer_config_fn: TrainerConfigFn, **kwargs) -> TrainerConfigFn:
    """Patches the trainer config produced by the trainer_config_fn."""

    def wrapped_fn():
        trainer_cfg = trainer_config_fn()
        trainer_cfg.set(**kwargs)
        return trainer_cfg

    return wrapped_fn


def chain_override(
    cfg: InstantiableConfig, config_modifiers: List[ModuleOverride]
) -> InstantiableConfig:
    """Produce a version of cfg_fn that applies each of `config_modifiers`.
    The modifiers are also set as an attribute for additional reusability / modularity.

    Args:
        cfg (TrainerConfig): the original config.
        config_modifiers (Callable): a list of functions that modify the config.
                                    The functions are called in order.

    Returns:
        TrainerConfigFn: a new config function that applies the modifiers.
    """
    for modifier in config_modifiers:
        cfg = modifier(cfg)
    return cfg
