"""Multi-language factory stub - delegates to coding-testing service."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MultiLanguageFactory:
    """Stub multi-language factory."""

    async def detect_language(self, *args: Any, **kwargs: Any) -> str:
        return "unknown"

    async def get_processor_for_file(self, file_path: Path) -> Any | None:
        """Return a processor for a given file path.

        This stub implementation returns None — actual processing is delegated
        to the coding-testing service. Callers should treat a None result as
        "no processor available" and fall back to simple node counting.
        """
        del file_path
        return None


async def get_multi_language_factory() -> MultiLanguageFactory:
    """Get multi-language factory instance."""
    return MultiLanguageFactory()
