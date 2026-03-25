"""Basic tests to ensure the package is properly installed."""

import importlib


def test_package_import() -> None:
    """Test that the package can be imported."""
    package_name = "eba_regulatory_agent"
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
