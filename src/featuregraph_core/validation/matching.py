import pandas as pd


def match_breaths(detected_indices, expert_indices, tolerance_samples):
    """
    Matches each expert-marked breath sample index to the nearest
    detected cycle's boundary index, within a tolerance window. Works
    for any two lists of sample indices, so this is dataset-agnostic.

    recall = fraction of expert-marked breaths that had a detected cycle
             boundary nearby (did the algorithm miss real breaths?)
    precision = fraction of detected cycle boundaries that had an
             expert-marked breath nearby (did the algorithm hallucinate
             extra cycles?)
    """
    detected = pd.DataFrame({'detected_index': sorted(detected_indices)}).astype(float)
    expert = pd.DataFrame({'expert_index': sorted(expert_indices)}).astype(float)

    if len(detected) == 0 or len(expert) == 0:
        return {
            'recall': float('nan'), 'precision': float('nan'),
            'n_expert': len(expert), 'n_detected': len(detected),
        }

    matched_forward = pd.merge_asof(
        expert, detected,
        left_on='expert_index', right_on='detected_index',
        direction='nearest', tolerance=tolerance_samples,
    )
    recall = matched_forward['detected_index'].notna().mean()

    matched_backward = pd.merge_asof(
        detected, expert,
        left_on='detected_index', right_on='expert_index',
        direction='nearest', tolerance=tolerance_samples,
    )
    precision = matched_backward['expert_index'].notna().mean()

    return {
        'recall': recall, 'precision': precision,
        'n_expert': len(expert), 'n_detected': len(detected),
    }
