"""Deterministic split helpers for dataset registration."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def stratified_validation_indices(
    labels: Sequence[int],
    validation_per_class: int,
    seed: int,
) -> set[int]:
    """Select a fixed number of validation indices per class.

    The returned indices refer to the original upstream ordering. The caller can
    then record a derived VisionLab split without changing sample identity.
    """

    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if not labels:
        raise ValueError("labels must not be empty")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    selected: set[int] = set()
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < validation_per_class:
            raise ValueError(
                f"class {label} has {len(indices)} samples, "
                f"cannot select {validation_per_class}"
            )
        rng = random.Random(f"{seed}:{label}")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected.update(shuffled[:validation_per_class])

    return selected
