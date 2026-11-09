"""Package-level smoke tests."""

from importlib.metadata import version

import symlmm


def test_version_matches_installed_metadata() -> None:
    assert symlmm.__version__ == version("symlmm")
