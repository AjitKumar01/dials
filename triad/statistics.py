"""Paired uncertainty conditional on fitted models and a fixed held-out split."""
import numpy as np


def paired_comparison(first, second, groups, *, bootstrap=2000, permutations=10000, seed=0):
    """Cluster bootstrap CI and paired cluster sign-flip test for a macro-mean difference.

    Duplicate complete bundles move together. Resampling preserves query multiplicity;
    the statistic is the query-weighted mean, not the unweighted mean of cluster means.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or len(first) != len(groups) or len(first) < 2:
        raise ValueError('Aligned nonempty one-dimensional metric vectors and groups are required')
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError('Metrics must be finite')
    if bootstrap < 1 or permutations < 1:
        raise ValueError('Positive resampling counts are required')
    difference = first - second
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, bootstrap, 128):
        sampled = rng.integers(0, len(sums), size=(min(128, bootstrap - start), len(sums)))
        estimates.extend((sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)).tolist())
    extremes = 0
    for start in range(0, permutations, 128):
        signs = 2 * rng.integers(0, 2, size=(min(128, permutations - start), len(sums)), dtype=np.int8) - 1
        null = (signs * sums).sum(axis=1) / len(difference)
        extremes += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    return {'delta': observed, 'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'p_value': (extremes + 1) / (permutations + 1), 'queries': len(first),
            'clusters': len(sums), 'bootstrap': bootstrap, 'permutations': permutations,
            'query_win_fraction': float(np.mean(difference > 1e-15)), 'seed': seed}


def holm_adjust(records, planned_tests=None):
    """In-place Holm correction; reserve the complete planned family for partial reports."""
    family = len(records) if planned_tests is None else planned_tests
    if family < len(records):
        raise ValueError('Planned test family is smaller than observed tests')
    previous = 0.0
    for rank, record in enumerate(sorted(records, key=lambda value: value['p_value'])):
        previous = min(1.0, max(previous, (family - rank) * record['p_value']))
        record.update(p_holm=previous, reject_05=previous < .05)
