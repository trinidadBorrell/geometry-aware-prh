"""Thin wrappers around scikit-learn splitters.

Indices are drawn once and stored as sample ids, so both endpoints receive the
same assignment. ``GroupShuffleSplit`` test sizes count groups. A validation
split is a second, separate call on the training subset only, and groups are
not allowed to cross train, validation, and test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from rho_cca.records import PairedDataset, Split


def _assignment(dataset: PairedDataset, train: Sequence[int], validation: Sequence[int], test: Sequence[int], *, seed, method: str) -> Split:
    labels = np.full(len(dataset.sample_ids), "", dtype=object)
    for name, positions in (("train", train), ("validation", validation), ("test", test)):
        for position in positions:
            labels[int(position)] = name
    named = [i for i, label in enumerate(labels) if label]
    return Split(
        train_ids=tuple(dataset.sample_ids[int(i)] for i in train),
        validation_ids=tuple(dataset.sample_ids[int(i)] for i in validation),
        test_ids=tuple(dataset.sample_ids[int(i)] for i in test),
        group_by_sample=tuple(
            (dataset.sample_ids[int(i)], dataset.group_ids[int(i)]) for i in named
        ),
        seed=None if seed is None else int(seed),
        fingerprint=dataset.fingerprint(),
        method=method,
    )


def split_rows(
    dataset: PairedDataset,
    *,
    test_size: float | int,
    validation_size: float | int | None = None,
    seed: int,
    stratify: Sequence | None = None,
) -> Split:
    """``train_test_split`` on independent rows. ``validation_size`` splits the training remainder."""
    index = np.arange(len(dataset.sample_ids))
    labels = None if stratify is None else np.asarray(stratify)
    if labels is not None and len(labels) != len(index):
        raise ValueError("stratify labels must match the dataset rows")
    train, test = train_test_split(index, test_size=test_size, random_state=seed, stratify=labels)
    validation = np.asarray([], dtype=np.int64)
    if validation_size is not None:
        train_labels = None if labels is None else labels[train]
        train, validation = train_test_split(
            train, test_size=validation_size, random_state=seed, stratify=train_labels
        )
    return _assignment(dataset, train, validation, test, seed=seed, method="train_test_split")


def split_groups(
    dataset: PairedDataset,
    *,
    test_size: float | int,
    validation_size: float | int | None = None,
    seed: int,
) -> Split:
    """``GroupShuffleSplit``. Sizes count groups. Validation is split from the training groups."""
    index = np.arange(len(dataset.sample_ids))
    groups = np.asarray(dataset.group_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(index, groups=groups))
    validation = np.asarray([], dtype=np.int64)
    if validation_size is not None:
        second = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed + 1)
        kept, held = next(second.split(train, groups=groups[train]))
        validation = train[held]
        train = train[kept]
    return _assignment(dataset, train, validation, test, seed=seed, method="GroupShuffleSplit")


def stratified_group_folds(
    dataset: PairedDataset,
    y: Sequence,
    *,
    n_splits: int,
    seed: int,
) -> list[Split]:
    """One ``StratifiedGroupKFold`` assignment per fold. Validation ids are empty."""
    labels = np.asarray(y)
    if len(labels) != len(dataset.sample_ids):
        raise ValueError("stratified group labels must match the dataset rows")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    index = np.arange(len(dataset.sample_ids))
    groups = np.asarray(dataset.group_ids)
    folds: list[Split] = []
    for fold, (train, test) in enumerate(splitter.split(index, labels, groups)):
        folds.append(
            _assignment(
                dataset,
                train,
                np.asarray([], dtype=np.int64),
                test,
                seed=seed,
                method=f"StratifiedGroupKFold[{fold}/{n_splits}]",
            )
        )
    return folds
