"""Name-based model registry and factory for config-driven construction."""
from typing import Callable

_MODELS: dict[str, Callable] = {}


def register(name: str):
    def deco(cls):
        if name in _MODELS:
            raise KeyError(f"model '{name}' already registered")
        _MODELS[name] = cls
        return cls

    return deco


def build_model(cfg: dict):
    name = cfg["name"]
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'; registered: {sorted(_MODELS)}")
    return _MODELS[name](**cfg.get("args", {}))
