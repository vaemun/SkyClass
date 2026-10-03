"""SDSS DR17 acquisition and photometry feature construction."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
BANDS = ("u", "g", "r", "i", "z")
CLASSES = ("STAR", "GALAXY", "QSO")


def _select_sql(class_name: str, limit: int) -> str:
    columns = ["p.objid", "p.ra", "p.dec", "s.class", "s.z AS redshift"]
    for prefix in ("psfMag", "psfMagErr", "modelMag", "modelMagErr", "extinction"):
        columns.extend(f"p.{prefix}_{band}" for band in BANDS)
    return (
        f"SELECT TOP {int(limit)} {', '.join(columns)} "
        "FROM SpecObj s JOIN PhotoObj p ON s.bestobjid = p.objid "
        f"WHERE s.class = '{class_name}' AND s.zWarning = 0 "
        "ORDER BY CHECKSUM(s.specobjid), s.specobjid"
    )


def fetch_sdss(cache_path: str | Path, limit_per_class: int = 33_333, refresh: bool = False) -> pd.DataFrame:
    """Download and cache the raw DR17 query result, or reuse the existing cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    if limit_per_class < 1:
        raise ValueError("limit_per_class must be positive")
    frames = []
    queries = []
    for class_name in CLASSES:
        query = _select_sql(class_name, limit_per_class)
        response = requests.get(
            SKYSERVER_SQL_URL,
            params={"cmd": query, "format": "csv"},
            headers={"User-Agent": "SkyClass/1.0 (SDSS DR17 photometric classification)"},
            timeout=300,
        )
        response.raise_for_status()
        if "#Table1" not in response.text:
            raise RuntimeError(f"SkyServer returned an unexpected response: {response.text[:500]}")
        class_frame = pd.read_csv(StringIO(response.text), comment="#")
        if class_frame.empty:
            raise RuntimeError(f"SkyServer returned no rows for class {class_name}")
        frames.append(class_frame)
        queries.append(query)
    frame = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    cache_path.with_suffix(".query.json").write_text(
        json.dumps({"endpoint": SKYSERVER_SQL_URL, "sampling": "class-capped deterministic CHECKSUM(specobjid) ordering", "limit_per_class": limit_per_class, "queries": queries}, indent=2),
        encoding="utf-8",
    )
    return frame


def prepare_photometry(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate magnitudes, deredden both systems, and derive colour diagnostics."""
    frame = raw.copy()
    magnitude_columns = [
        f"{prefix}_{band}"
        for prefix in ("psfMag", "psfMagErr", "modelMag", "modelMagErr", "extinction")
        for band in BANDS
    ]
    missing = [column for column in magnitude_columns if column not in frame]
    if missing:
        raise ValueError(f"Required SDSS columns missing: {', '.join(missing)}")

    frame[magnitude_columns] = frame[magnitude_columns].apply(pd.to_numeric, errors="coerce")
    flux_magnitudes = frame[[f"{kind}_{band}" for kind in ("psfMag", "modelMag") for band in BANDS]]
    sentinel_rows = flux_magnitudes.eq(-9999).any(axis=1)
    nonfinite_rows = ~np.isfinite(flux_magnitudes.to_numpy()).all(axis=1)
    errors = frame[[f"{kind}_{band}" for kind in ("psfMagErr", "modelMagErr") for band in BANDS]]
    valid_errors = np.isfinite(errors.to_numpy()).all(axis=1) & (errors.to_numpy() >= 0).all(axis=1)
    extinctions = frame[[f"extinction_{band}" for band in BANDS]]
    valid_extinctions = np.isfinite(extinctions.to_numpy()).all(axis=1)
    valid_labels = frame["class"].isin(CLASSES).to_numpy()
    keep = ~sentinel_rows.to_numpy() & ~nonfinite_rows & valid_errors & valid_extinctions & valid_labels

    report = {
        "input_rows": int(len(frame)),
        "sentinel_magnitude_rows": int(sentinel_rows.sum()),
        "nonfinite_magnitude_rows": int(nonfinite_rows.sum()),
        "invalid_or_missing_error_rows": int((~valid_errors).sum()),
        "invalid_or_missing_extinction_rows": int((~valid_extinctions).sum()),
        "unsupported_or_missing_class_rows": int((~valid_labels).sum()),
        "dropped_rows": int((~keep).sum()),
        "retained_rows": int(keep.sum()),
    }
    frame = frame.loc[keep].copy()

    for band in BANDS:
        frame[f"psf_{band}"] = frame[f"psfMag_{band}"] - frame[f"extinction_{band}"]
        frame[f"model_{band}"] = frame[f"modelMag_{band}"] - frame[f"extinction_{band}"]
    color_pairs = {
        "u-g": ("u", "g"), "g-r": ("g", "r"), "r-i": ("r", "i"), "i-z": ("i", "z"),
        "u-r": ("u", "r"), "g-i": ("g", "i"), "r-z": ("r", "z"), "u-z": ("u", "z"),
    }
    for color, (first, second) in color_pairs.items():
        frame[f"psf_color_{color}"] = frame[f"psf_{first}"] - frame[f"psf_{second}"]
        frame[f"model_color_{color}"] = frame[f"model_{first}"] - frame[f"model_{second}"]
        frame[f"psf_color_err_{color}"] = np.sqrt(
            frame[f"psfMagErr_{first}"].pow(2) + frame[f"psfMagErr_{second}"].pow(2)
        )

    for band in BANDS:
        frame[f"concentration_{band}"] = frame[f"psf_{band}"] - frame[f"model_{band}"]
    return frame, report