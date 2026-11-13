"""Tests for processing/multi_language_factory.py."""

import pytest

from src.processing.multi_language_factory import MultiLanguageFactory, get_multi_language_factory


class TestMultiLanguageFactory:
    """Test the MultiLanguageFactory stub class."""

    def test_has_detect_language_method(self):
        """MultiLanguageFactory must expose a callable detect_language method."""
        factory = MultiLanguageFactory()
        assert callable(factory.detect_language)

    @pytest.mark.asyncio
    async def test_detect_language_returns_string(self):
        """detect_language must return a string (not None, not int, etc.)."""
        factory = MultiLanguageFactory()
        result = await factory.detect_language()
        assert isinstance(result, str)
        assert len(result) > 0, "detect_language should not return empty string"

    @pytest.mark.asyncio
    async def test_detect_language_stub_returns_unknown(self):
        """Stub implementation returns 'unknown' — verifies current contract."""
        factory = MultiLanguageFactory()
        result = await factory.detect_language()
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_detect_language_ignores_positional_args(self):
        """detect_language accepts arbitrary positional args without error."""
        factory = MultiLanguageFactory()
        result = await factory.detect_language("some_file.py", "/workspace")
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_detect_language_ignores_keyword_args(self):
        """detect_language accepts arbitrary keyword args without error."""
        factory = MultiLanguageFactory()
        result = await factory.detect_language(language_hint="python", file_path="x.py")
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_detect_language_idempotent(self):
        """Multiple calls should return the same result (no side effects)."""
        factory = MultiLanguageFactory()
        r1 = await factory.detect_language("a.py")
        r2 = await factory.detect_language("b.js")
        assert r1 == r2 == "unknown"


class TestGetMultiLanguageFactory:
    """Test the factory function."""

    @pytest.mark.asyncio
    async def test_returns_multi_language_factory_instance(self):
        """get_multi_language_factory must return a MultiLanguageFactory instance."""
        factory = await get_multi_language_factory()
        assert isinstance(factory, MultiLanguageFactory)

    @pytest.mark.asyncio
    async def test_returned_factory_has_working_detect_language(self):
        """Factory-produced instance should have a working detect_language."""
        factory = await get_multi_language_factory()
        result = await factory.detect_language()
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_returns_new_instance_each_call(self):
        """get_multi_language_factory returns a fresh instance each time (no singleton)."""
        f1 = await get_multi_language_factory()
        f2 = await get_multi_language_factory()
        assert f1 is not f2
