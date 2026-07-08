"""Visual Perception Engine package (lazy imports for optional lightweight tools)."""

from __future__ import annotations

__all__ = [
    "Engine",
    "ModelRegistry",
    "ModelExporter",
    "export_default_models",
    "ModelCard",
    "ModelHeadCard",
]


def __getattr__(name: str):
    if name == "Engine":
        from vp_engine.engine import Engine

        return Engine
    if name == "ModelRegistry":
        from model_management.registry import ModelRegistry

        return ModelRegistry
    if name == "ModelExporter":
        from model_management.model_exporter import ModelExporter

        return ModelExporter
    if name == "export_default_models":
        from model_management.model_exporter import export_default_models

        return export_default_models
    if name == "ModelCard":
        from model_management.model_cards import ModelCard

        return ModelCard
    if name == "ModelHeadCard":
        from model_management.model_cards import ModelHeadCard

        return ModelHeadCard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
