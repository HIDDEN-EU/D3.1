"""
KDEF dataset helper.

KDEF has structured per-image metadata (angle, gender, session) stored in the
manifest CSV.  This module provides a factory function that filters the manifest
DataFrame by the requested metadata and returns a :class:`ManifestDataset`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import ManifestDataset


def build_kdef_dataset(
    df: pd.DataFrame,
    transform=None,
    angles: Optional[list[str]] = None,
    genders: Optional[list[str]] = None,
    sessions: Optional[list[str]] = None,
) -> ManifestDataset:
    """Filter a KDEF manifest DataFrame and return a :class:`ManifestDataset`.

    Parameters
    ----------
    df :
        Rows already filtered to a single split (train / val / test).
    transform :
        Transform applied to each image.
    angles :
        Subset of ``["FL", "HL", "S", "HR", "FR"]`` to keep.
        ``None`` keeps all angles.
    genders :
        Subset of ``["F", "M"]`` to keep.  ``None`` keeps both.
    sessions :
        Subset of ``["A", "B"]`` to keep.  ``None`` keeps both.
    """
    if angles is not None:
        df = df[df["angle"].isin(angles)]
    if genders is not None:
        df = df[df["gender"].isin(genders)]
    if sessions is not None:
        df = df[df["session"].isin(sessions)]

    records = list(zip(df["path"], df["label"].astype(int)))
    return ManifestDataset(records, transform)
