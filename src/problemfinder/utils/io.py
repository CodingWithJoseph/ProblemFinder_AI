"""I/O helpers for reading and writing dataframes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV into a normalised pandas DataFrame."""

    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).fillna("")
    df.reset_index(drop=True, inplace=True)
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist ``df`` to ``path`` as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
