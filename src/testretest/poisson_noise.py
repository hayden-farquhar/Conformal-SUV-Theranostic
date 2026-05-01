"""Poisson noise injection for synthetic test-retest reference.

Pre-registered fallback (§3.5): if <50 real serial pairs available,
generate synthetic test-retest replicates by injecting Poisson noise
at reduced dose levels.

Methodology:
1. Convert SUV map to raw counts using acquisition parameters
2. Scale counts to simulate dose reduction (50%, 25%, 10%)
3. Draw Poisson samples from the reduced counts
4. Convert back to SUV
5. Re-extract lesion features from noisy volume
6. Compute within-lesion CV across replicates

References:
    Schaefferkoetter et al. (2020)
    Lodge et al. (2011) JNM 52(1):137-144

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.5)
"""

from __future__ import annotations

import numpy as np


# Pre-registered dose reduction levels (§3.5)
DOSE_REDUCTION_LEVELS = [0.50, 0.25, 0.10]
N_REPLICATES = 10


def suv_to_counts(
    suv_volume: np.ndarray,
    injected_dose_bq: float,
    patient_weight_kg: float,
    decay_factor: float,
    frame_duration_sec: float,
    calibration_factor: float = 1.0,
) -> np.ndarray:
    """Convert SUV volume back to approximate raw counts.

    The inverse of the SUV calculation:
    counts ≈ SUV × (dose × decay / weight) × frame_duration × calibration

    This is an approximation — the exact relationship depends on
    scanner-specific reconstruction parameters that we don't have.
    The Poisson noise model is valid as long as the count-rate is
    proportional to activity, which holds for clinical PET.

    Parameters
    ----------
    suv_volume : np.ndarray
        3D SUV image.
    injected_dose_bq : float
        Injected dose in Bq.
    patient_weight_kg : float
        Patient weight in kg.
    decay_factor : float
        2^(-uptake_time / half_life).
    frame_duration_sec : float
        PET frame duration in seconds.
    calibration_factor : float
        Scanner calibration factor (typically 1.0 if unknown).

    Returns
    -------
    np.ndarray
        Estimated counts (float, not yet Poisson-sampled).
    """
    weight_g = patient_weight_kg * 1000.0
    dose_corrected = injected_dose_bq * decay_factor

    # Activity concentration = SUV × (dose / weight)
    activity = suv_volume * (dose_corrected / weight_g)

    # Counts ≈ activity × frame_duration × calibration
    counts = activity * frame_duration_sec * calibration_factor

    # Ensure non-negative
    counts = np.maximum(counts, 0.0)

    return counts


def inject_poisson_noise(
    counts: np.ndarray,
    dose_fraction: float,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Inject Poisson noise at a reduced dose level.

    Parameters
    ----------
    counts : np.ndarray
        Original count image (at full dose).
    dose_fraction : float
        Fraction of original dose (e.g. 0.50 for 50% dose).
    rng : RandomState, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy count image (integer-valued but float dtype).
    """
    if rng is None:
        rng = np.random.RandomState()

    # Scale counts to reduced dose
    reduced_counts = counts * dose_fraction

    # Draw Poisson samples
    # For very high counts, use normal approximation to avoid overflow
    noisy = np.zeros_like(reduced_counts)
    high_count_mask = reduced_counts > 1e6
    low_count_mask = ~high_count_mask

    if low_count_mask.any():
        noisy[low_count_mask] = rng.poisson(
            np.maximum(reduced_counts[low_count_mask], 0).astype(np.float64)
        )

    if high_count_mask.any():
        # Normal approximation: N(λ, √λ)
        lam = reduced_counts[high_count_mask]
        noisy[high_count_mask] = np.maximum(
            0, rng.normal(lam, np.sqrt(lam))
        )

    return noisy


def counts_to_suv(
    counts: np.ndarray,
    injected_dose_bq: float,
    patient_weight_kg: float,
    decay_factor: float,
    frame_duration_sec: float,
    dose_fraction: float,
    calibration_factor: float = 1.0,
) -> np.ndarray:
    """Convert noisy counts back to SUV.

    Parameters
    ----------
    counts : np.ndarray
        Noisy count image.
    dose_fraction : float
        The dose fraction used to generate these counts.
        Needed to correctly scale back to the original dose level.
    [other params same as suv_to_counts]

    Returns
    -------
    np.ndarray
        SUV volume reconstructed from noisy counts.
    """
    weight_g = patient_weight_kg * 1000.0
    dose_corrected = injected_dose_bq * decay_factor

    # Activity = counts / (frame_duration × calibration)
    activity = counts / (frame_duration_sec * calibration_factor)

    # Correct for the dose reduction (we want SUV at the original dose)
    activity = activity / dose_fraction

    # SUV = activity / (dose / weight)
    suv = activity / (dose_corrected / weight_g)

    return suv


def generate_noisy_replicates(
    suv_volume: np.ndarray,
    injected_dose_bq: float,
    patient_weight_kg: float,
    decay_factor: float,
    frame_duration_sec: float,
    dose_fractions: list[float] = None,
    n_replicates: int = N_REPLICATES,
    seed: int = 42,
) -> dict[float, list[np.ndarray]]:
    """Generate multiple noisy SUV replicates at each dose level.

    Parameters
    ----------
    suv_volume : np.ndarray
        Original SUV volume.
    dose_fractions : list[float]
        Dose reduction levels (default: [0.50, 0.25, 0.10]).
    n_replicates : int
        Number of replicates per dose level (default: 10).
    seed : int
        Random seed.

    Returns
    -------
    dict[float, list[np.ndarray]]
        Mapping dose_fraction -> list of noisy SUV volumes.
    """
    if dose_fractions is None:
        dose_fractions = DOSE_REDUCTION_LEVELS

    rng = np.random.RandomState(seed)
    counts = suv_to_counts(
        suv_volume, injected_dose_bq, patient_weight_kg,
        decay_factor, frame_duration_sec,
    )

    replicates = {}
    for frac in dose_fractions:
        replicates[frac] = []
        for _ in range(n_replicates):
            noisy_counts = inject_poisson_noise(counts, frac, rng)
            noisy_suv = counts_to_suv(
                noisy_counts, injected_dose_bq, patient_weight_kg,
                decay_factor, frame_duration_sec, frac,
            )
            replicates[frac].append(noisy_suv)

    return replicates


def compute_replicate_cv(
    suv_values: list[float],
) -> float:
    """Compute CV from multiple SUV measurements of the same lesion.

    Parameters
    ----------
    suv_values : list[float]
        SUV measurements across replicates.

    Returns
    -------
    float
        Coefficient of variation (%) = SD/mean × 100.
    """
    arr = np.array(suv_values)
    mean = arr.mean()
    if mean <= 0:
        return 0.0
    return float(arr.std() / mean * 100.0)
