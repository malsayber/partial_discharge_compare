from __future__ import annotations

"""Orchestrate preprocessing of raw sessions."""

from argparse import ArgumentParser
from pathlib import Path
import logging
from typing import Iterable

import numpy as np

import config
from preprocess import discovery, io, cleaning, augmentation, windowing

logger = logging.getLogger(__name__)


def process_session(session: discovery.Session, dataset: str, force: bool, adv_denoise: bool, augment: bool) -> None:
    """Process a single recording session."""
    for sensor, path in session.sensor_files.items():
        signal_path = Path(path)
        if signal_path.suffix == ".csv":
            x = io.load_pd_csv(signal_path)
        else:
            x = io.load_pd_hdf5(signal_path)

        fs = 1.0  # placeholder sampling rate

        window_id_base = f"{session.cable_id}_{sensor}"

        x_bp = cleaning.bandpass_filter(
            x,
            config.CONFIG.preprocessing_options.bandpass_hz[0],
            config.CONFIG.preprocessing_options.bandpass_hz[1],
            fs,
        )
        cleaning.save_cleaned_signal(x_bp, session.cable_id, "standard_denoising_normalisation", f"{window_id_base}_bp")

        if adv_denoise:
            x_den = cleaning.advanced_denoise(x_bp)
            method_dir = "advanced_denoising/VMD"
            cleaning.save_cleaned_signal(x_den, session.cable_id, method_dir, f"{window_id_base}_den")
        else:
            x_den = x_bp

        x_norm = cleaning.zscore_normalize(x_den)
        cleaning.save_cleaned_signal(x_norm, session.cable_id, "standard_denoising_normalisation", f"{window_id_base}_norm")

        if augment:
            x_aug = augmentation.time_warp(x_norm)
            x_aug = augmentation.add_jitter(x_aug)
        else:
            x_aug = x_norm

        windows = windowing.segment_signal(
            x_aug,
            config.CONFIG.preprocessing_options.window_length_ms,
            fs,
        )
        labels = windowing.load_window_labels(session.label_file, len(windows))

        out_dir = config.PROCESSED_DIR / session.cable_id / sensor
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (win, label) in enumerate(zip(windows, labels)):
            out_path = out_dir / f"{window_id_base}_{idx}.npy"
            if out_path.exists() and not force:
                continue
            np.save(out_path, win)
            if label is not None:
                with open(out_path.with_suffix(".label"), "w", encoding="utf-8") as fh:
                    fh.write(str(label))


def process_npy_file(record: discovery.FileRecord, force: bool, adv_denoise: bool, augment: bool) -> None:
    """Process a single raw ``.npy`` file from a station."""
    x = io.load_pd_npy(record.file_path)
    fs = 1.0  # placeholder sampling rate
    fid = Path(record.file_path).stem

    x_bp = cleaning.bandpass_filter(
        x,
        config.CONFIG.preprocessing_options.bandpass_hz[0],
        config.CONFIG.preprocessing_options.bandpass_hz[1],
        fs,
    )
    cleaning.save_cleaned_signal(x_bp, record.station_id, "standard_denoising_normalisation", f"{fid}_bp")

    if adv_denoise:
        x_den = cleaning.advanced_denoise(x_bp)
        method_dir = "advanced_denoising/VMD"
        cleaning.save_cleaned_signal(x_den, record.station_id, method_dir, f"{fid}_den")
    else:
        x_den = x_bp

    x_norm = cleaning.zscore_normalize(x_den)
    cleaning.save_cleaned_signal(x_norm, record.station_id, "standard_denoising_normalisation", f"{fid}_norm")

    if augment:
        x_aug = augmentation.time_warp(x_norm)
        x_aug = augmentation.add_jitter(x_aug)
    else:
        x_aug = x_norm

    windows = windowing.segment_signal(
        x_aug,
        config.CONFIG.preprocessing_options.window_length_ms,
        fs,
    )

    out_dir = config.PROCESSED_DIR / record.station_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, win in enumerate(windows):
        out_path = out_dir / f"{fid}_{idx}.npy"
        if out_path.exists() and not force:
            continue
        np.save(out_path, win)


def run(dataset: str, force: bool, adv_denoise: bool, augment: bool) -> None:
    """Run preprocessing for ``dataset``."""
    records = discovery.discover_npy_files(dataset)
    if records:
        for rec in records:
            process_npy_file(rec, force, adv_denoise, augment)
        return

    sessions = discovery.discover_sessions(dataset)
    for session in sessions:
        process_session(session, dataset, force, adv_denoise, augment)


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess PD datasets")
    parser.add_argument("dataset", help="Dataset name under raw data dir")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--advanced-denoise", action="store_true", help="Use advanced denoising")
    parser.add_argument("--augment", action="store_true", help="Augment training data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.dataset, args.force, args.advanced_denoise, args.augment)
