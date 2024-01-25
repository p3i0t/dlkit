from typing import Callable

import torch.nn as nn
from dlkit.models.bases import ModelConfig
from dlkit.models.gpt import GPT, GPTConfig
from dlkit.models.lstm import LSTM, LSTMConfig

MODEL = "model"
ARCHITECTURE = "architecture"
KNOWN_CATEGORIES = [MODEL, ARCHITECTURE]

__all__ = ["get_model", "list_all_models", "list_all_model_classes"]


# the value is a model class only; to be initialized with a proper config
_global_model_class_registry = {}

# each value is a pair (model_class, model_config)
_global_model_registry = {}


def register_model_class(name: str, model: Callable):
    if name in _global_model_class_registry:
        raise ValueError(f"Cannot register duplicate model ({name})")
    _global_model_class_registry[name] = model


def register_model(name: str, model: Callable, arch_config: ModelConfig):
    # if name not in _global_model_registry:
    #     raise ValueError(
    #         f"Cannot register architecture for unknown model ({name})")
    if name in _global_model_registry:
        raise ValueError(f"Cannot register duplicate model ({name})")

    _global_model_registry[name] = (model, arch_config)


def get_model(
    name: str = "GPT_small", d_in: int = None, d_out: int = None
) -> nn.Module:
    """Return a initialized model"""
    o = _global_model_registry.get(name, None)
    if o is None:
        raise ValueError(f"model {name} is not registered.")
    model, config = o
    config.d_in = d_in
    config.d_out = d_out
    return model(config)


def get_model_class(name: str) -> Callable:
    """Return a model class (callable), to be initialized with config."""
    model = _global_model_class_registry.get(name, None)
    if model is None:
        raise ValueError(f"model class {name} is not registered.")
    return model


def list_all_models() -> list[str]:
    """Get the list of all registered models.

    Returns:
        list[str]: list of model names.
    """
    return list(_global_model_registry.keys())


def list_all_model_classes() -> list[Callable]:
    """Get the list of all registered model classes.

    Returns:
        list[Callable]: list of model classes, who are callable to initialize models.
    """
    return list(_global_model_class_registry.keys())


register_model_class(name="GPT", model=GPT)
register_model_class(name="LSTM", model=LSTM)


register_model(
    name="GPT_small",
    model=GPT,
    arch_config=GPTConfig(
        d_model=512,
        n_head=4,
        n_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        max_positions=256,
    ),
)


register_model(
    name="GPT_medium",
    model=GPT,
    arch_config=GPTConfig(
        d_model=512,
        n_head=4,
        n_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_positions=256,
    ),
)

register_model(
    name="GPT_large",
    model=GPT,
    arch_config=GPTConfig(
        d_model=512,
        n_head=8,
        n_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        max_positions=256,
    ),
)

register_model(
    name="LSTM_small",
    model=LSTM,
    arch_config=LSTMConfig(
        d_model=512,
        n_layers=2,
        dropout=0.1,
    ),
)

register_model(
    name="LSTM_medium",
    model=LSTM,
    arch_config=LSTMConfig(
        d_model=512,
        n_layers=4,
        dropout=0.1,
    ),
)
