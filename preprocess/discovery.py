from __future__ import annotations

"""Discovery utilities for raw PD datasets."""

from collections import namedtuple
from pathlib import Path
from typing import List

from .sklearn_data import ensure_dataset

import config

Session = namedtuple("Session", ["cable_id", "sensor_files", "label_file"])


def discover_sessions(dataset: str) -> List[Session]:
    """Find all recording sessions in ``data/raw/{dataset}``.

    Parameters
    ----------
    dataset
        Dataset name under the raw data directory.

    Returns
    -------
    List[Session]
        Detected sessions with sensor paths and label files.
    """
    ensure_dataset(dataset)
    base = config.RAW_DIR / dataset
    sessions: List[Session] = []
    if not base.exists():
        return sessions

    for cable_dir in base.iterdir():
        if not cable_dir.is_dir():
            continue
        sensor_files = {}
        for file in cable_dir.glob("*.csv"):
            sensor_files[file.stem.upper()] = str(file)
        for file in cable_dir.glob("*.h5"):
            sensor_files[file.stem.upper()] = str(file)
        label_file = next(cable_dir.glob("labels.*"), None)
        sessions.append(
            Session(
                cable_id=cable_dir.name,
                sensor_files=sensor_files,
                label_file=str(label_file) if label_file else None,
            )
        )
    return sessions
