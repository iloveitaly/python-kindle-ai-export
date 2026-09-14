"""Test python-kindle-ai-export."""

import python_kindle_ai_export


def test_import() -> None:
    """Test that the  can be imported."""
    assert isinstance(python_kindle_ai_export.__name__, str)


def test_version() -> None:
    """Test that the version is available."""
    assert isinstance(python_kindle_ai_export.__version__, str)