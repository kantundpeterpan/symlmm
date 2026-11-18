"""Package-level smoke tests."""

import importlib.util
from importlib.metadata import version

import symlmm


def test_version_matches_installed_metadata() -> None:
    assert symlmm.__version__ == version("symlmm")


def test_canonical_module_is_removed():
    assert importlib.util.find_spec("symlmm.canonical") is None


def test_blocks_module_is_public():
    from symlmm import blocks

    assert hasattr(blocks, "Block")
