"""Model specifications: grouping factors and designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    """A single grouping factor."""

    name: str

    @property
    def label(self) -> str:
        """Return the symbol label for this grouping factor."""
        return self.name


@dataclass(frozen=True)
class Nested:
    """A factor nested within another factor."""

    child: str
    within: str

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.child

    @property
    def label(self) -> str:
        """Return the composite symbol label for this grouping factor."""
        return f"{self.within}:{self.child}"


@dataclass(frozen=True)
class Crossed:
    """A crossed combination of factors."""

    left: str
    right: str

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.right

    @property
    def label(self) -> str:
        """Return the composite symbol label for this grouping factor."""
        return f"{self.left}:{self.right}"
