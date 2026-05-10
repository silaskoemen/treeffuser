from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treeffuser import Treeffuser


@dataclass(frozen=True)
class Variant:
    name: str
    params: dict[str, Any]

    def make_model(self, seed: int) -> Treeffuser:
        params = dict(self.params)
        params["seed"] = seed
        return Treeffuser(**params)


def make_variants(config: list[dict[str, Any]]) -> list[Variant]:
    variants = []
    for item in config:
        if not item.get("enabled", True):
            continue
        name = item["name"]
        params = item.get("params", {})
        variants.append(Variant(name=name, params=params))
    if not variants:
        raise ValueError("Benchmark config must enable at least one variant.")
    return variants

