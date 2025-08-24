"""XAI package initializer

Ленивая обертка для стабильного импорта функции run_xai_analysis:
from xai import run_xai_analysis
"""

from typing import Any


def run_xai_analysis(*args: Any, **kwargs: Any):  # noqa: D401
    """Thin lazy-loader wrapper around xai.xai_integration.run_xai_analysis."""
    from .xai_integration import run_xai_analysis as _impl  # импорт при первом вызове
    return _impl(*args, **kwargs)


__all__ = ["run_xai_analysis"]


