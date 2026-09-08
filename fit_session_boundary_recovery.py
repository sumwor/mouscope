"""Fit exponential recovery following behavioral-session boundaries.

This standalone diagnostic imports the established read-only behavioral loading
and reward definitions. It does not modify behavioral source data or pipeline
code.
"""

from __future__ import annotations

import matplotlib

# Keep imports of the existing behavioral pipeline safe in headless runs.
_original_matplotlib_use = matplotlib.use


def _headless_matplotlib_use(backend, *args, **kwargs):
    if str(backend).lower() in {"qtagg", "qt5agg", "qt6agg"}:
        backend = "Agg"
    return _original_matplotlib_use(backend, *args, **kwargs)


matplotlib.use = _headless_matplotlib_use
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analyze_session_boundary import (
    ReadOnlyBehDataOdor,
    load_protocol_sessions,
    rewarded_vector,
)


PARAMETER_LOWER = np.array([0.0, 0.0, 1.0])
PARAMETER_UPPER = np.array([1.0, 1.0, 300.0])
MIN_FIT_BINS = 8
PRE_BOUNDARY_TRIALS = 300
POST_BOUNDARY_TRIALS = 300
BIN_TRIALS = 20
INITIAL_POST_TRIALS = 50
PERCENT_BASELINE_ATOL = 1e-12


def exponential_recovery(n, p_inf, amplitude, recovery_lambda):
    """Evaluate P(n) = P_inf - A exp(-n/lambda)."""

    n = np.asarray(n, dtype=float)
    return p_inf - amplitude * np.exp(-n / recovery_lambda)


def binned_model_prediction(bin_starts, bin_ends, parameters):
    """Average the trial-level model over each inclusive integer bin."""

    p_inf, amplitude, recovery_lambda = np.asarray(parameters, dtype=float)
    return np.array(
        [
            exponential_recovery(
                np.arange(int(start), int(end) + 1),
                p_inf,
                amplitude,
                recovery_lambda,
            ).mean()
            for start, end in zip(bin_starts, bin_ends)
        ],
        dtype=float,
    )


def parameter_warnings(parameters, fit_success):
    """Return explicit near-bound and fitted-P(0) warning flags."""

    if not fit_success or not np.all(np.isfinite(parameters)):
        return {
            "PInfWarning": False,
            "AWarning": False,
            "LambdaWarning": False,
            "P0Warning": False,
            "BoundaryWarning": True,
        }

    p_inf, amplitude, recovery_lambda = np.asarray(parameters, dtype=float)
    p_inf_warning = bool(p_inf <= 0.01 or p_inf >= 0.99)
    amplitude_warning = bool(amplitude <= 0.01 or amplitude >= 0.99)
    lambda_margin = 0.01 * (PARAMETER_UPPER[2] - PARAMETER_LOWER[2])
    lambda_warning = bool(
        recovery_lambda <= PARAMETER_LOWER[2] + lambda_margin
        or recovery_lambda >= PARAMETER_UPPER[2] - lambda_margin
    )
    p0_warning = bool(p_inf - amplitude < 0.0 or p_inf - amplitude > 1.0)
    return {
        "PInfWarning": p_inf_warning,
        "AWarning": amplitude_warning,
        "LambdaWarning": lambda_warning,
        "P0Warning": p0_warning,
        "BoundaryWarning": bool(
            p_inf_warning or amplitude_warning or lambda_warning or p0_warning
        ),
    }


def _failed_fit(n_fit_bins, message):
    result = {
        "P_inf": np.nan,
        "A": np.nan,
        "lambda": np.nan,
        "P0": np.nan,
        "RSS": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "FitSuccess": False,
        "OptimizerMessage": message,
        "NFitBins": int(n_fit_bins),
    }
    result.update(parameter_warnings(np.full(3, np.nan), False))
    return result


def _initial_parameters(observed):
    late = float(np.nanmean(observed[-min(3, len(observed)) :]))
    p_inf_starts = np.clip(
        np.array([late - 0.1, late, late + 0.1]),
        PARAMETER_LOWER[0],
        PARAMETER_UPPER[0],
    )
    return [
        np.array([p_inf, amplitude, recovery_lambda], dtype=float)
        for p_inf in p_inf_starts
        for amplitude in (0.05, 0.2, 0.5)
        for recovery_lambda in (10.0, 75.0, 200.0)
    ]


def fit_exponential_recovery(curve, min_bins=MIN_FIT_BINS):
    """Fit the exponential model to finite binned observations."""

    required = {"BinStart", "BinEnd", "ObservedMean"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"Curve is missing required columns: {sorted(missing)}")

    starts = pd.to_numeric(curve["BinStart"], errors="coerce").to_numpy(float)
    ends = pd.to_numeric(curve["BinEnd"], errors="coerce").to_numpy(float)
    observed = pd.to_numeric(
        curve["ObservedMean"], errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(observed)
    starts = starts[finite]
    ends = ends[finite]
    observed = observed[finite]
    n_fit_bins = len(observed)
    if n_fit_bins < min_bins:
        return _failed_fit(n_fit_bins, f"Fewer than {min_bins} finite bins")

    best = None
    for initial in _initial_parameters(observed):
        try:
            result = least_squares(
                lambda parameters: binned_model_prediction(
                    starts, ends, parameters
                )
                - observed,
                x0=initial,
                bounds=(PARAMETER_LOWER, PARAMETER_UPPER),
                method="trf",
            )
        except (ValueError, FloatingPointError):
            continue
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        residual = binned_model_prediction(starts, ends, result.x) - observed
        rss = float(np.sum(residual**2))
        if np.isfinite(rss) and (best is None or rss < best[0]):
            best = (rss, result)

    if best is None:
        return _failed_fit(n_fit_bins, "No successful finite optimizer result")

    rss, optimizer = best
    parameters = optimizer.x
    rmse = float(np.sqrt(rss / n_fit_bins))
    centered = observed - observed.mean()
    total_sum_squares = float(np.sum(centered**2))
    r2 = (
        float(1.0 - rss / total_sum_squares)
        if not np.isclose(total_sum_squares, 0.0)
        else np.nan
    )
    fit = {
        "P_inf": float(parameters[0]),
        "A": float(parameters[1]),
        "lambda": float(parameters[2]),
        "P0": float(parameters[0] - parameters[1]),
        "RSS": rss,
        "RMSE": rmse,
        "R2": r2,
        "FitSuccess": True,
        "OptimizerMessage": str(optimizer.message),
        "NFitBins": int(n_fit_bins),
    }
    fit.update(parameter_warnings(parameters, True))
    return fit


def _drop_metrics(baseline, initial_post):
    absolute_drop = float(baseline - initial_post)
    percent_valid = bool(
        np.isfinite(baseline)
        and np.isfinite(initial_post)
        and not np.isclose(
            baseline, 0.0, rtol=0.0, atol=PERCENT_BASELINE_ATOL
        )
    )
    percent_drop = (
        float(100.0 * absolute_drop / baseline) if percent_valid else np.nan
    )
    return absolute_drop, percent_drop, percent_valid


def extract_boundary_data(
    session_frames,
    pre_trials=PRE_BOUNDARY_TRIALS,
    post_trials=POST_BOUNDARY_TRIALS,
    bin_trials=BIN_TRIALS,
):
    """Create post-boundary bin rows and empirical boundary metrics."""

    bin_rows = []
    metric_rows = []
    for boundary_number, (previous, next_session) in enumerate(
        zip(session_frames[:-1], session_frames[1:]), start=1
    ):
        if (
            previous.empty
            or next_session.empty
            or "reward" not in previous
            or "reward" not in next_session
        ):
            continue
        previous_rewarded = rewarded_vector(previous)[-pre_trials:]
        next_rewarded = rewarded_vector(next_session)[:post_trials]
        if not len(previous_rewarded) or not len(next_rewarded):
            continue

        metadata = {
            "BoundaryNumber": boundary_number,
            "PreviousSession": int(previous["_session_number"].iloc[-1]),
            "NextSession": int(next_session["_session_number"].iloc[0]),
            "PreviousSessionIndex": int(previous["_session_index"].iloc[-1]),
            "NextSessionIndex": int(next_session["_session_index"].iloc[0]),
            "PreviousDate": previous["_date"].iloc[-1],
            "NextDate": next_session["_date"].iloc[0],
            "PreviousProtocolDay": previous["_protocol_day"].iloc[-1],
            "NextProtocolDay": next_session["_protocol_day"].iloc[0],
        }
        baseline = float(np.mean(previous_rewarded))
        initial_post = float(
            np.mean(next_rewarded[: min(INITIAL_POST_TRIALS, len(next_rewarded))])
        )
        absolute_drop, percent_drop, percent_valid = _drop_metrics(
            baseline, initial_post
        )
        metric_rows.append(
            metadata
            | {
                "NPreBaselineTrials": len(previous_rewarded),
                "NPostTrials": len(next_rewarded),
                "PreBoundaryBaseline": baseline,
                "InitialPostPerformance": initial_post,
                "DropMagnitude": absolute_drop,
                "AbsoluteDrop": absolute_drop,
                "PercentDrop": percent_drop,
                "PercentDropValid": percent_valid,
            }
        )

        for start in range(0, len(next_rewarded), bin_trials):
            values = next_rewarded[start : start + bin_trials]
            end = start + len(values) - 1
            bin_rows.append(
                metadata
                | {
                    "BinStart": start,
                    "BinEnd": end,
                    "BinCenter": (start + end) / 2.0,
                    "NTrials": len(values),
                    "Performance": float(np.mean(values)),
                }
            )

    return pd.DataFrame(bin_rows), pd.DataFrame(metric_rows)


def aggregate_group_curve(boundary_bins):
    """Aggregate bin performance over boundaries, preserving boundary weighting."""

    if boundary_bins.empty:
        return pd.DataFrame()
    group_columns = ["Protocol", "Genotype", "BinStart"]
    return (
        boundary_bins.groupby(group_columns, dropna=False, sort=True)
        .agg(
            BinEnd=("BinEnd", "max"),
            BinCenter=("BinCenter", "max"),
            ObservedMean=("Performance", "mean"),
            ObservedSEM=("Performance", "sem"),
            NBoundaries=("Performance", "size"),
            NAnimals=("Animal", "nunique"),
        )
        .reset_index()
    )


def _empirical_animal_summary(metrics):
    percent_valid = metrics["PercentDropValid"].astype(bool)
    valid_percent = pd.to_numeric(
        metrics.loc[percent_valid, "PercentDrop"], errors="coerce"
    )
    return {
        "NBoundaries": int(len(metrics)),
        "MeanPreBoundaryBaseline": float(metrics["PreBoundaryBaseline"].mean()),
        "MedianPreBoundaryBaseline": float(
            metrics["PreBoundaryBaseline"].median()
        ),
        "MeanDropMagnitude": float(metrics["DropMagnitude"].mean()),
        "MedianDropMagnitude": float(metrics["DropMagnitude"].median()),
        "MeanAbsoluteDrop": float(metrics["AbsoluteDrop"].mean()),
        "MedianAbsoluteDrop": float(metrics["AbsoluteDrop"].median()),
        "NPercentDropValid": int(valid_percent.notna().sum()),
        "MeanPercentDrop": float(valid_percent.mean()),
        "MedianPercentDrop": float(valid_percent.median()),
    }


def fit_animal_curves(boundary_bins, boundary_metrics):
    """Pool each animal's boundaries by bin and fit its recovery curve."""

    rows = []
    grouping = boundary_metrics.groupby(["Animal", "Protocol"], sort=True)
    for (animal, protocol), metrics in grouping:
        animal_bins = boundary_bins[
            (boundary_bins["Animal"] == animal)
            & (boundary_bins["Protocol"] == protocol)
        ]
        curve = (
            animal_bins.groupby("BinStart", sort=True)
            .agg(
                BinEnd=("BinEnd", "max"),
                ObservedMean=("Performance", "mean"),
            )
            .reset_index()
        )
        fit = fit_exponential_recovery(curve)
        first_bin = animal_bins.iloc[0] if not animal_bins.empty else None
        rows.append(
            {
                "Animal": animal,
                "Protocol": protocol,
                "Genotype": first_bin["Genotype"] if first_bin is not None else np.nan,
                "Gender": first_bin["Gender"] if first_bin is not None else np.nan,
            }
            | fit
            | _empirical_animal_summary(metrics)
        )
    return pd.DataFrame(rows)
