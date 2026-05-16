from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

DEFAULT_N_FOLDS = 6
TUNING_FOLD = 0


@dataclass(frozen=True)
class FoldSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class TuningSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    name: str
    master_seed: int
    n_folds: int
    fold_assignment: np.ndarray
    X: Float[np.ndarray, "n x_dim"]
    y: Float[np.ndarray, "n y_dim"]

    @property
    def tuning(self) -> TuningSplit:
        val_idx = np.flatnonzero(self.fold_assignment == TUNING_FOLD)
        train_idx = np.flatnonzero(self.fold_assignment != TUNING_FOLD)
        return TuningSplit(train_idx=train_idx, val_idx=val_idx)

    @property
    def eval_folds(self) -> list[FoldSplit]:
        """One FoldSplit per non-tuning fold.

        Each eval fold tests on itself and trains on every other fold including the
        tuning fold. No leakage: the tuning objective only ever scored fold 0, so
        every test fold here was unseen during hyperparameter selection.
        """
        return [
            FoldSplit(
                train_idx=np.flatnonzero(self.fold_assignment != i),
                test_idx=np.flatnonzero(self.fold_assignment == i),
            )
            for i in range(self.n_folds)
            if i != TUNING_FOLD
        ]

    def slice(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]


def build_splits(
    X: Float[np.ndarray, "n x_dim"],
    y: Float[np.ndarray, "n y_dim"],
    master_seed: int,
    n_folds: int = DEFAULT_N_FOLDS,
    name: str = "",
) -> DatasetSplits:
    """Deterministic K-fold assignment from a single master_seed.

    Fold 0 is the tuning fold; folds 1..n_folds-1 are eval folds. Each eval fold
    trains on all other folds (including fold 0) and tests on itself.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y row counts differ: {X.shape[0]} vs {y.shape[0]}")
    if n_folds < 3:
        raise ValueError(f"Need at least 3 folds (1 tuning + 2 eval), got {n_folds}")
    n = X.shape[0]
    if n < n_folds * 2:
        raise ValueError(f"Need at least {n_folds * 2} samples for K={n_folds}, got {n}")

    rng = np.random.default_rng(master_seed)
    perm = rng.permutation(n)
    fold_assignment = np.empty(n, dtype=np.int64)
    fold_assignment[perm] = np.arange(n) % n_folds
    return DatasetSplits(
        name=name,
        master_seed=master_seed,
        n_folds=n_folds,
        fold_assignment=fold_assignment,
        X=X,
        y=y,
    )
