from __future__ import annotations

import pandas as pd

from ..utils._wearable import (
    load_wearable_data_constraints,
    load_wearable_eda,
    load_wearable_hr,
    load_wearable_stress_levels,
    load_wearable_subject_info,
    load_wearable_tags,
    load_wearable_temp,
)

# No wearable_map exists yet in ..utils._rename_map, unlike bidmc_map
# and eastman_map -- column names below come directly from the utils
# loader as-is. If a standardized FeatureGraph naming convention should
# apply here too, add a wearable_map to _rename_map.py and these
# wrappers can apply it the same way bidmc()/eastman() do.


def wearable_temp(subject: str = "S01", *, refresh: bool = False) -> pd.DataFrame:
    """Load skin temperature for one wearable-dataset subject."""
    return load_wearable_temp(subject, refresh=refresh)


def wearable_eda(subject: str = "S01", *, refresh: bool = False) -> pd.DataFrame:
    """Load electrodermal activity for one wearable-dataset subject."""
    return load_wearable_eda(subject, refresh=refresh)


def wearable_hr(subject: str = "S01", *, refresh: bool = False) -> pd.DataFrame:
    """Load heart rate for one wearable-dataset subject."""
    return load_wearable_hr(subject, refresh=refresh)


def wearable_tags(
    subject: str = "S01",
    *,
    reference_start_timestamp: float | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load protocol-boundary button-press tags for one subject."""
    return load_wearable_tags(
        subject,
        reference_start_timestamp=reference_start_timestamp,
        refresh=refresh,
    )


def wearable_subject_info(*, refresh: bool = False) -> pd.DataFrame:
    """Load demographic info for all wearable-dataset subjects."""
    return load_wearable_subject_info(refresh=refresh)


def wearable_stress_levels(protocol_version: int = 1, *, refresh: bool = False) -> pd.DataFrame:
    """Load self-reported stress levels for one protocol version (1 or 2)."""
    return load_wearable_stress_levels(protocol_version, refresh=refresh)


def wearable_data_constraints(*, refresh: bool = False) -> str:
    """Return the dataset authors' documented data constraints verbatim."""
    return load_wearable_data_constraints(refresh=refresh)