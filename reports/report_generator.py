"""Utilities for creating MNE reports from saved images."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from mne import Report

logger = logging.getLogger(__name__)


def generate_report(image_paths: Iterable[str | Path], output_file: str | Path) -> None:
    """Create an MNE report stacking all images.

    Parameters
    ----------
    image_paths : Iterable[str | Path]
        Paths to image files to include in the report.
    output_file : str | Path
        Path where the HTML report will be saved.
    """
    logger.info("Generating MNE report...")
    report = Report(title="Partial Discharge Classification Report")
    for img in image_paths:
        report.add_image(str(img), title=Path(img).stem)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path), overwrite=True)
    logger.info("Report saved to %s", output_file)
