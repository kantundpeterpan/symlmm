"""Resolve factor declarations into immutable factors with full ancestry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError
from .specs import Crossed, Factor, FactorSpec, Group, Nested


class FactorResolver:
    """Resolve factor declarations and string references to `Factor` objects."""

    def __init__(
        self,
        specs: Sequence[FactorSpec],
        observations: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        self.specs = tuple(specs)
        self.observations = observations
        self._resolved: dict[str, Factor] = {}
        self._by_label: dict[str, Factor] = {}

    def resolve_all(self) -> tuple[Factor, ...]:
        """Resolve every declaration and return factors in declaration order."""
        for spec in self.specs:
            self._resolve(spec, ())
        return tuple(self._resolved.values())

    def resolve(self, ref: str | FactorSpec | Factor | tuple[str, ...]) -> Factor:
        """Resolve a name, component path, declaration, or factor."""
        if isinstance(ref, Factor):
            return ref
        if isinstance(ref, str):
            try:
                return self._resolved[ref]
            except KeyError as exc:
                raise ModelError(f"unknown factor: {ref!r}") from exc
        if isinstance(ref, tuple):
            for factor in self._resolved.values():
                if factor.components == ref:
                    return factor
            raise ModelError(f"unknown factor path: {ref!r}")
        return self._resolve(ref, ())

    def _resolve(self, spec: FactorSpec, stack: tuple[str, ...]) -> Factor:
        if isinstance(spec, Group):
            return self._register(
                Factor(
                    name=spec.name,
                    label=spec.name,
                    components=(spec.name,),
                    index=spec.index,
                    parent=None,
                    kind="group",
                )
            )
        if isinstance(spec, Nested):
            if spec.child in stack:
                raise ModelError(f"cycle in nested factors: {' -> '.join(stack + (spec.child,))}")
            ancestor = spec.within if isinstance(spec.within, str) else spec.within.name
            if ancestor == spec.child:
                raise ModelError(f"cycle in nested factors: {spec.child} -> {spec.child}")
            parent = self._resolve_reference(spec.within, stack + (spec.child,), exclude=spec)
            if parent is not None:
                path = (*parent.components, spec.child)
            elif isinstance(spec.within, str):
                path = (spec.within, spec.child)
            else:
                raise ModelError("Nested ancestor could not be resolved")
            return self._register(
                Factor(
                    name=spec.child,
                    label=":".join(path),
                    components=path,
                    index=self._resolved_index(spec, path),
                    parent=parent,
                    kind="nested",
                )
            )
        if isinstance(spec, Crossed):
            parts: list[str] = []
            for component in spec.components:
                resolved = self._resolve_reference(component, stack, exclude=spec)
                if resolved is None:
                    if not isinstance(component, str):
                        raise ModelError("Crossed component could not be resolved")
                    parts.append(component)
                else:
                    parts.extend(resolved.components)
            factor = Factor(
                name=spec.name,
                label=":".join(parts),
                components=tuple(parts),
                index=self._resolved_index(spec, tuple(parts)),
                parent=None,
                kind="crossed",
            )
            return self._register(factor)
        raise ModelError(f"unsupported factor declaration: {spec!r}")

    def _resolved_index(self, spec: FactorSpec, components: tuple[str, ...]) -> Any:
        """Return the explicit index, or a default index built from full ancestry."""
        default = sp.Symbol(spec.label, integer=True)
        if spec.index == default:
            return sp.Symbol(":".join(components), integer=True)
        return spec.index

    def _resolve_reference(
        self,
        ref: str | FactorSpec,
        stack: tuple[str, ...],
        exclude: FactorSpec | None = None,
    ) -> Factor | None:
        if not isinstance(ref, str):
            return self._resolve(ref, stack)
        if ref in self._resolved:
            return self._resolved[ref]
        for spec in self.specs:
            if spec is exclude:
                continue
            if spec.name == ref:
                return self._resolve(spec, stack)
        return None

    def _register(self, factor: Factor) -> Factor:
        existing = self._resolved.get(factor.name)
        if existing is not None:
            if existing.components != factor.components:
                raise ModelError(
                    f"conflicting grouping-factor specs for {factor.name!r}: "
                    f"{existing.label!r} and {factor.label!r}"
                )
            if existing.index != factor.index or existing.kind != factor.kind:
                raise ModelError(
                    f"conflicting grouping-factor specs for {factor.name!r}: "
                    f"{existing!r} and {factor!r}"
                )
            return existing
        existing_label = self._by_label.get(factor.label)
        if existing_label is not None:
            raise ModelError(
                f"conflicting factor label {factor.label!r}: "
                f"{existing_label.name!r} and {factor.name!r}"
            )
        self._resolved[factor.name] = factor
        self._by_label[factor.label] = factor
        return factor
