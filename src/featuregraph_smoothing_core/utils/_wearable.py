from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import requests

# Source: Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025).
# "Wearable Device Dataset from Induced Stress and Structured Exercise
# Sessions." PhysioNet. https://doi.org/10.13026/he0v-tf17
#
# Directory structure, subject-ID list, and the two-header-row signal
# format below were confirmed directly against PhysioNet's own file
# listing and documentation page -- not assumed. One exception: the
# exact byte-level layout of tags.csv is described only in prose in
# PhysioNet's docs, not independently verified here against a real
# sample row. _parse_tags() validates its own assumption and raises
# loudly if the values it reads don't look like plausible timestamps,
# rather than silently producing shifted data if that assumption is
# wrong.
WEARABLE_DATASET_SLUG = "wearable-device-dataset"
WEARABLE_VERSION = "1.0.1"
WEARABLE_ROOT_URL = f"https://physionet.org/files/{WEARABLE_DATASET_SLUG}/{WEARABLE_VERSION}"
WEARABLE_STRESS_BASE_URL = f"{WEARABLE_ROOT_URL}/Wearable_Dataset/STRESS"

# Version-1 (original protocol) and version-2 (revised protocol)
# participant IDs, confirmed against the dataset's actual directory
# listing. f14 is split into two recordings after a Bluetooth
# disconnection (per the dataset's own data_constraints.txt) and is
# NOT recombined here -- see the physionet_wearable_protocol_study.md
# note that this is deliberately left for a future explicit
# composition study rather than silently repaired.
V1_SUBJECTS: tuple[str, ...] = tuple(f"S{n:02d}" for n in range(1, 19))
V2_SUBJECTS: tuple[str, ...] = (
    tuple(f"f{n:02d}" for n in range(1, 14))
    + ("f14_a", "f14_b")
    + tuple(f"f{n:02d}" for n in range(15, 19))
)
VALID_SUBJECTS: tuple[str, ...] = V1_SUBJECTS + V2_SUBJECTS

# Signal kinds with a confirmed, implemented parser. BVP, ACC, and IBI
# exist in the dataset and are not needed by any current FeatureGraph
# study, so their row-level format is intentionally left unimplemented
# below rather than guessed at.
ImplementedKind = Literal["TEMP", "EDA", "HR"]
UNIMPLEMENTED_KINDS = ("BVP", "ACC", "IBI")

_SIGNAL_COLUMN_NAMES: dict[str, str] = {
    "TEMP": "temp_celsius",
    "EDA": "eda_microsiemens",
    "HR": "heart_rate_bpm",
}

TOP_LEVEL_FILES = (
    "subject-info.csv",
    "Stress_Level_v1.csv",
    "Stress_Level_v2.csv",
    "data_constraints.txt",
    "README.txt",
    "LICENSE.txt",
)


def get_cache_dir() -> Path:
    """Return the wearable-dataset cache directory outside the Git repo."""
    cache_dir = (
        Path.home()
        / ".cache"
        / "featuregraph"
        / "physionet_wearable"
        / WEARABLE_VERSION
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _validate_subject(subject: str) -> None:
    if not isinstance(subject, str):
        raise TypeError("subject must be a string")
    if subject not in VALID_SUBJECTS:
        raise ValueError(
            f"Unknown subject {subject!r}. Valid subjects are "
            f"{V1_SUBJECTS[0]}-{V1_SUBJECTS[-1]} (version 1) and "
            f"{V2_SUBJECTS[0]}-{V2_SUBJECTS[-1]} including the split "
            "f14_a/f14_b (version 2)."
        )


def _download(
    url: str,
    destination: Path,
    *,
    refresh: bool,
    timeout: int,
) -> Path:
    """Shared download-to-cache logic used by both subject and
    top-level file downloads."""
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")

        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def download_wearable_subject_file(
    subject: str,
    kind: str,
    *,
    refresh: bool = False,
    timeout: int = 60,
) -> Path:
    """Download one raw per-subject file into the external cache.

    This downloads the raw bytes for any kind present in the dataset
    (including TOP_LEVEL_FILES' per-subject counterparts like BVP, ACC,
    IBI) -- download support is format-agnostic. Only TEMP, EDA, and HR
    have an implemented row-level parser below.
    """
    _validate_subject(subject)
    filename = f"{kind}.csv"
    destination = get_cache_dir() / subject / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WEARABLE_STRESS_BASE_URL}/{subject}/{filename}"
    return _download(url, destination, refresh=refresh, timeout=timeout)


def download_wearable_top_level_file(
    filename: str,
    *,
    refresh: bool = False,
    timeout: int = 60,
) -> Path:
    """Download one dataset-level (non-per-subject) file."""
    if filename not in TOP_LEVEL_FILES:
        raise ValueError(
            f"Unknown top-level file {filename!r}. Expected one of "
            f"{TOP_LEVEL_FILES}."
        )
    destination = get_cache_dir() / filename
    url = f"{WEARABLE_ROOT_URL}/{filename}"
    return _download(url, destination, refresh=refresh, timeout=timeout)


def _parse_utc_timestamp(raw: str) -> float:
    """Parse a timestamp field as either a raw Unix epoch or a
    formatted UTC datetime string, returning Unix seconds either way.

    Confirmed against a real file: this dataset's row-1 session start
    time is a formatted string (e.g. '2013-02-20 17:55:29'), not a raw
    epoch number as PhysioNet's prose documentation seemed to imply.
    The old-looking year is expected, not an error -- the dataset's own
    de-identification note states session dates were shifted by over a
    year.
    """
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return pd.Timestamp(raw, tz="UTC").timestamp()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse {raw!r} as either a Unix timestamp or a "
            "recognizable UTC datetime string."
        ) from exc


def _parse_single_column_empatica_csv(
    path: Path,
    *,
    column_name: str,
) -> pd.DataFrame:
    """Parse a raw Empatica export with two metadata header rows.

    Row 1: session start time (UTC; Unix epoch or formatted datetime).
    Row 2: sample rate (Hz).
    Remaining rows: one value per row, at that fixed rate.
    """
    with path.open("r") as file:
        start_timestamp = _parse_utc_timestamp(file.readline())
        sample_rate_hz = float(file.readline().strip())
        values = [float(line.strip()) for line in file if line.strip()]

    if sample_rate_hz <= 0:
        raise ValueError(
            f"Non-positive sample rate ({sample_rate_hz}) read from {path}"
        )

    seconds_from_start = [index / sample_rate_hz for index in range(len(values))]
    df = pd.DataFrame(
        {
            "seconds_from_start": seconds_from_start,
            "timestamp_utc": [start_timestamp + s for s in seconds_from_start],
            column_name: values,
        }
    )
    df.attrs["start_timestamp_utc"] = start_timestamp
    df.attrs["sample_rate_hz"] = sample_rate_hz
    return df


def _parse_tags(path: Path, *, reference_start_timestamp: float | None = None) -> pd.DataFrame:
    """Parse tags.csv: one button-press Unix UTC timestamp per row, no
    header rows (per PhysioNet's documentation of this dataset).

    This assumption about the row format is NOT independently verified
    against a raw byte sample -- only against prose documentation.
    A sanity check below raises loudly if the values read don't look
    like plausible Unix timestamps, rather than silently returning
    shifted data if the assumption is wrong.
    """
    with path.open("r") as file:
        timestamps = [_parse_utc_timestamp(line) for line in file if line.strip()]

    implausible = [t for t in timestamps if not (1_000_000_000 < t < 4_000_000_000)]
    if implausible:
        raise RuntimeError(
            "tags.csv contained values that don't look like plausible Unix "
            f"timestamps: {implausible[:5]}{'...' if len(implausible) > 5 else ''}. "
            "This likely means tags.csv has a different row format than "
            f"assumed (zero header rows) -- inspect {path} directly before "
            "trusting this result."
        )

    df = pd.DataFrame(
        {
            "tag_index": range(len(timestamps)),
            "timestamp_utc": timestamps,
        }
    )
    if reference_start_timestamp is not None:
        df["seconds_from_reference"] = df["timestamp_utc"] - reference_start_timestamp
    return df


def _load_signal(
    subject: str,
    kind: ImplementedKind,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    if kind not in _SIGNAL_COLUMN_NAMES:
        if kind in UNIMPLEMENTED_KINDS:
            raise NotImplementedError(
                f"{kind} exists in this dataset but its row-level format "
                "is not yet implemented here (not needed by any current "
                "FeatureGraph study). download_wearable_subject_file() can "
                "still fetch the raw bytes if you want to inspect it "
                "yourself before adding a parser."
            )
        raise ValueError(f"Unknown signal kind {kind!r}")

    path = download_wearable_subject_file(subject, kind, refresh=refresh)
    df = _parse_single_column_empatica_csv(path, column_name=_SIGNAL_COLUMN_NAMES[kind])
    df["subject"] = subject
    df.attrs["subject"] = subject
    df.attrs["kind"] = kind
    df.attrs["source_file"] = str(path)
    df.attrs["wearable_version"] = WEARABLE_VERSION
    return df


def load_wearable_temp(subject: str, *, refresh: bool = False) -> pd.DataFrame:
    """Load skin temperature (°C) for one subject."""
    return _load_signal(subject, "TEMP", refresh=refresh)


def load_wearable_eda(subject: str, *, refresh: bool = False) -> pd.DataFrame:
    """Load electrodermal activity (µS) for one subject."""
    return _load_signal(subject, "EDA", refresh=refresh)


def load_wearable_hr(subject: str, *, refresh: bool = False) -> pd.DataFrame:
    """Load heart rate (bpm) for one subject."""
    return _load_signal(subject, "HR", refresh=refresh)


def load_wearable_tags(
    subject: str,
    *,
    reference_start_timestamp: float | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load button-press protocol-boundary tags for one subject.

    tags.csv timestamps are absolute Unix UTC, on the same clock as the
    other signal files' declared start times -- NOT relative to session
    start on their own. Pass reference_start_timestamp (e.g. from
    ``load_wearable_hr(subject).attrs["start_timestamp_utc"]``) to also
    get a seconds-from-reference column; this is left for the caller to
    supply deliberately rather than the loader silently picking one
    signal's start time as "the" reference.
    """
    _validate_subject(subject)
    path = download_wearable_subject_file(subject, "tags", refresh=refresh)
    df = _parse_tags(path, reference_start_timestamp=reference_start_timestamp)
    df["subject"] = subject
    df.attrs["subject"] = subject
    df.attrs["source_file"] = str(path)
    df.attrs["wearable_version"] = WEARABLE_VERSION
    return df


def load_wearable_subject_info(*, refresh: bool = False) -> pd.DataFrame:
    """Load demographic info (age, weight, height) for all subjects."""
    path = download_wearable_top_level_file("subject-info.csv", refresh=refresh)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.attrs["source_file"] = str(path)
    df.attrs["wearable_version"] = WEARABLE_VERSION
    return df


def load_wearable_stress_levels(
    protocol_version: Literal[1, 2],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load self-reported stress levels for one protocol version."""
    if protocol_version not in (1, 2):
        raise ValueError("protocol_version must be 1 or 2")
    filename = f"Stress_Level_v{protocol_version}.csv"
    path = download_wearable_top_level_file(filename, refresh=refresh)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.attrs["source_file"] = str(path)
    df.attrs["protocol_version"] = protocol_version
    df.attrs["wearable_version"] = WEARABLE_VERSION
    return df


def load_wearable_data_constraints(*, refresh: bool = False) -> str:
    """Return the dataset authors' own data_constraints.txt verbatim.

    Returned as plain text rather than parsed into structured
    exclusions -- the authors' own wording of a constraint shouldn't be
    silently reinterpreted into a different shape here.
    """
    path = download_wearable_top_level_file("data_constraints.txt", refresh=refresh)
    return path.read_text()


def clear_wearable_cache() -> None:
    """Remove all cached wearable-dataset files."""
    cache_dir = get_cache_dir()
    for path in sorted(cache_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()