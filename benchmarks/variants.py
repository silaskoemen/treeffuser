from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.baselines import make_baseline_model
from treeffuser import Treeffuser


@dataclass(frozen=True)
class Variant:
    name: str
    params: dict[str, Any]
    model: str = "treeffuser"

    def make_model(self, seed: int) -> Treeffuser:
        if self.model != "treeffuser":
            return make_baseline_model(model_type=self.model, params=self.params, seed=seed)
        params = dict(self.params)
        params["seed"] = seed
        return Treeffuser(**params)


def make_variants(config: list[dict[str, Any]]) -> list[Variant]:
    variants = []
    for item in config:
        if not item.get("enabled", True):
            continue
        name = item["name"]
        model = item.get("model", "treeffuser")
        params = item.get("params", {})
        variants.append(Variant(name=name, params=params, model=model))
    if not variants:
        raise ValueError("Benchmark config must enable at least one variant.")
    return variants
