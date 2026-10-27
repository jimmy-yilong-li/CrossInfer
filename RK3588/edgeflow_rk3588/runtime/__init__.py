"""Shared reference-runtime interfaces, without board-specific producers."""

__all__ = [
    "VisualPayload",
]

_LAZY_EXPORTS = {
    "VisualPayload": "visual_payload",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module = import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
