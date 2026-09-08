"""Fit exponential recovery following behavioral-session boundaries.

This standalone diagnostic imports the established read-only behavioral loading
and reward definitions. It does not modify behavioral source data or pipeline
code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
import matplotlib.pyplot as plt
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
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_BOOTSTRAP_SEED = 240513
GROUP_ORDER = (("AB", "WT"), ("AB", "HET"), ("CD", "WT"), ("CD", "HET"))
GROUP_COLORS = {
    ("AB", "WT"): "tab:blue",
    ("AB", "HET"): "tab:cyan",
    ("CD", "WT"): "tab:orange",
    ("CD", "HET"): "tab:red",
}


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


def resample_animal_clusters(group_bins, rng, sampled_animals=None):
    """Resample animals and retain every boundary row per selected cluster."""

    animals = np.asarray(pd.unique(group_bins["Animal"]))
    if not len(animals):
        return group_bins.assign(BootstrapCluster=pd.Series(dtype=int))
    if sampled_animals is None:
        sampled_animals = rng.choice(animals, size=len(animals), replace=True)
    clusters = []
    for cluster_number, animal in enumerate(sampled_animals):
        cluster = group_bins[group_bins["Animal"] == animal].copy()
        cluster["BootstrapCluster"] = cluster_number
        clusters.append(cluster)
    return pd.concat(clusters, ignore_index=True) if clusters else pd.DataFrame()


def bootstrap_group_fit(group_bins, n_replicates, rng):
    """Cluster-bootstrap animals while retaining boundary-weighted group means."""

    parameters = []
    curves = []
    trial_index = np.arange(POST_BOUNDARY_TRIALS, dtype=float)
    for _ in range(n_replicates):
        sample = resample_animal_clusters(group_bins, rng)
        curve = aggregate_group_curve(sample)
        fit = fit_exponential_recovery(curve)
        parameter_vector = np.array(
            [fit["P_inf"], fit["A"], fit["lambda"]], dtype=float
        )
        if fit["FitSuccess"] and np.all(np.isfinite(parameter_vector)):
            parameters.append(parameter_vector)
            curves.append(exponential_recovery(trial_index, *parameter_vector))

    n_successful = len(parameters)
    summary = {
        "NBootstrapRequested": int(n_replicates),
        "NBootstrapSuccessful": int(n_successful),
        "BootstrapSuccessFraction": (
            float(n_successful / n_replicates) if n_replicates else np.nan
        ),
    }
    if n_successful:
        parameter_array = np.asarray(parameters)
        curve_array = np.asarray(curves)
        for column_index, name in enumerate(("P_inf", "A", "lambda")):
            low, high = np.percentile(
                parameter_array[:, column_index], [2.5, 97.5]
            )
            summary[f"{name}_CI_low"] = float(low)
            summary[f"{name}_CI_high"] = float(high)
        band = np.percentile(curve_array, [2.5, 97.5], axis=0).T
    else:
        for name in ("P_inf", "A", "lambda"):
            summary[f"{name}_CI_low"] = np.nan
            summary[f"{name}_CI_high"] = np.nan
        band = np.full((POST_BOUNDARY_TRIALS, 2), np.nan)
    return summary, band


def _empirical_group_summary(metrics):
    valid = metrics["PercentDropValid"].astype(bool)
    percent = pd.to_numeric(metrics.loc[valid, "PercentDrop"], errors="coerce")
    return {
        "MeanPreBoundaryBaseline": float(metrics["PreBoundaryBaseline"].mean()),
        "MedianPreBoundaryBaseline": float(
            metrics["PreBoundaryBaseline"].median()
        ),
        "MeanInitialPostPerformance": float(
            metrics["InitialPostPerformance"].mean()
        ),
        "MedianInitialPostPerformance": float(
            metrics["InitialPostPerformance"].median()
        ),
        "MeanDropMagnitude": float(metrics["DropMagnitude"].mean()),
        "MedianDropMagnitude": float(metrics["DropMagnitude"].median()),
        "MeanAbsoluteDrop": float(metrics["AbsoluteDrop"].mean()),
        "MedianAbsoluteDrop": float(metrics["AbsoluteDrop"].median()),
        "NPercentDropValid": int(percent.notna().sum()),
        "MeanPercentDrop": float(percent.mean()),
        "MedianPercentDrop": float(percent.median()),
    }


def fit_group_curves(boundary_bins, boundary_metrics, n_replicates, seed):
    """Fit primary boundary-weighted group curves and cluster bootstrap CIs."""

    summary_rows = []
    curve_rows = []
    rng = np.random.default_rng(seed)
    for (protocol, genotype), group_bins in boundary_bins.groupby(
        ["Protocol", "Genotype"], sort=True
    ):
        group_metrics = boundary_metrics[
            (boundary_metrics["Protocol"] == protocol)
            & (boundary_metrics["Genotype"] == genotype)
        ]
        curve = aggregate_group_curve(group_bins)
        fit = fit_exponential_recovery(curve)
        bootstrap, band = bootstrap_group_fit(group_bins, n_replicates, rng)
        n_animals = int(group_bins["Animal"].nunique())
        n_boundaries = int(len(group_metrics))
        summary_rows.append(
            {
                "Protocol": protocol,
                "Genotype": genotype,
                "NAnimals": n_animals,
                "NBoundaries": n_boundaries,
            }
            | fit
            | bootstrap
            | _empirical_group_summary(group_metrics)
        )

        curve = curve.copy()
        if fit["FitSuccess"]:
            parameters = np.array([fit["P_inf"], fit["A"], fit["lambda"]])
            curve["FittedBinMean"] = binned_model_prediction(
                curve["BinStart"].to_numpy(),
                curve["BinEnd"].to_numpy(),
                parameters,
            )
            curve["FittedAtCenter"] = exponential_recovery(
                curve["BinCenter"].to_numpy(), *parameters
            )
        else:
            curve["FittedBinMean"] = np.nan
            curve["FittedAtCenter"] = np.nan
        trial_index = np.arange(POST_BOUNDARY_TRIALS)
        curve["FittedCurveCI_low"] = np.interp(
            curve["BinCenter"], trial_index, band[:, 0]
        )
        curve["FittedCurveCI_high"] = np.interp(
            curve["BinCenter"], trial_index, band[:, 1]
        )
        curve_rows.extend(curve.to_dict("records"))

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def plot_protocol_recovery(group_summary, group_curve, protocol, output_path):
    """Plot observed group means/SEM with exponential fits and bootstrap bands."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    annotations = []
    for genotype in ("WT", "HET"):
        key = (protocol, genotype)
        color = GROUP_COLORS[key]
        curve = group_curve[
            (group_curve["Protocol"] == protocol)
            & (group_curve["Genotype"].astype(str).str.upper() == genotype)
        ].sort_values("BinStart")
        fit_rows = group_summary[
            (group_summary["Protocol"] == protocol)
            & (group_summary["Genotype"].astype(str).str.upper() == genotype)
        ]
        if curve.empty or fit_rows.empty:
            continue
        fit = fit_rows.iloc[0]
        ax.errorbar(
            curve["BinCenter"],
            curve["ObservedMean"],
            yerr=curve["ObservedSEM"],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            linestyle="none",
            alpha=0.85,
            label=f"{genotype} observed mean +/- SEM",
        )
        if bool(fit["FitSuccess"]):
            x = np.arange(POST_BOUNDARY_TRIALS)
            fitted = exponential_recovery(
                x, fit["P_inf"], fit["A"], fit["lambda"]
            )
            ax.plot(
                x,
                fitted,
                color=color,
                linewidth=2.2,
                label=f"{genotype} exponential fit",
            )
            low = np.interp(
                x,
                curve["BinCenter"],
                curve["FittedCurveCI_low"],
                left=curve["FittedCurveCI_low"].iloc[0],
                right=curve["FittedCurveCI_low"].iloc[-1],
            )
            high = np.interp(
                x,
                curve["BinCenter"],
                curve["FittedCurveCI_high"],
                left=curve["FittedCurveCI_high"].iloc[0],
                right=curve["FittedCurveCI_high"].iloc[-1],
            )
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        warning_suffix = " WARNING" if bool(fit["BoundaryWarning"]) else ""
        annotations.append(
            f"{genotype}: A={fit['A']:.3f} "
            f"[{fit['A_CI_low']:.3f}, {fit['A_CI_high']:.3f}], "
            f"lambda={fit['lambda']:.1f} "
            f"[{fit['lambda_CI_low']:.1f}, {fit['lambda_CI_high']:.1f}]\n"
            f"N={int(fit['NAnimals'])} animals, {int(fit['NBoundaries'])} boundaries"
            f"{warning_suffix}"
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Post-boundary trial (zero-based)", fontsize=11)
    ax.set_ylabel("P(correct)", fontsize=11)
    ax.set_title(f"{protocol}: exponential post-session recovery", fontsize=15)
    ax.text(
        0.01,
        0.98,
        "animal-cluster bootstrap 95% CIs\n" + "\n".join(annotations),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _valid_animal_fits(animal_summary):
    return animal_summary[
        animal_summary["FitSuccess"].astype(bool)
        & ~animal_summary["BoundaryWarning"].astype(bool)
    ].copy()


def plot_parameter_comparison(animal_summary, output_path):
    """Show individual valid animal parameters with median and IQR."""

    valid = _valid_animal_fits(animal_summary)
    labels = [f"{protocol} {genotype}" for protocol, genotype in GROUP_ORDER]
    rng = np.random.default_rng(7301)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, parameter, ylabel in zip(
        axes,
        ("A", "lambda", "P_inf"),
        ("Fitted reset amplitude A", "Recovery timescale lambda (trials)", "P_inf"),
    ):
        for position, key in enumerate(GROUP_ORDER):
            protocol, genotype = key
            values = pd.to_numeric(
                valid.loc[
                    (valid["Protocol"] == protocol)
                    & (valid["Genotype"].astype(str).str.upper() == genotype),
                    parameter,
                ],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            jitter = rng.uniform(-0.08, 0.08, len(values))
            ax.scatter(
                position + jitter,
                values,
                s=28,
                alpha=0.7,
                color=GROUP_COLORS[key],
                edgecolor="white",
                linewidth=0.4,
            )
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            ax.vlines(position, q1, q3, color="black", linewidth=3, zorder=4)
            ax.scatter(
                position,
                median,
                marker="D",
                s=65,
                color="white",
                edgecolor="black",
                linewidth=1,
                zorder=5,
            )
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    excluded = int(len(animal_summary) - len(valid))
    fig.suptitle(
        "Animal-level recovery parameters: points with median/IQR\n"
        f"Valid non-warning fits shown; {excluded} failed or warning fits excluded",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_fitted_vs_empirical_drop(animal_summary, output_path):
    """Compare fitted A with empirical absolute and relative reset metrics."""

    valid = _valid_animal_fits(animal_summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key in GROUP_ORDER:
        protocol, genotype = key
        rows = valid[
            (valid["Protocol"] == protocol)
            & (valid["Genotype"].astype(str).str.upper() == genotype)
        ]
        axes[0].scatter(
            rows["A"],
            rows["MeanAbsoluteDrop"],
            color=GROUP_COLORS[key],
            s=40,
            alpha=0.75,
            label=f"{protocol} {genotype}",
        )
        axes[1].scatter(
            rows["A"],
            rows["MeanPercentDrop"],
            color=GROUP_COLORS[key],
            s=40,
            alpha=0.75,
            label=f"{protocol} {genotype}",
        )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Fitted A", fontsize=10)
    axes[0].set_ylabel("Mean empirical AbsoluteDrop", fontsize=10)
    axes[0].set_title("Same-unit comparison", fontsize=13)
    axes[1].set_xlim(left=0)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Fitted A", fontsize=10)
    axes[1].set_ylabel("Mean empirical PercentDrop (%)", fontsize=10)
    axes[1].set_title("Relative empirical reset", fontsize=13)
    for ax in axes:
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Fitted recovery amplitude versus empirical boundary drop\n"
        "Valid non-warning animal fits",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir, bootstrap_replicates, bootstrap_seed):
    text = f"""Session-boundary exponential recovery analysis

This standalone read-only diagnostic fits P(n) = P_inf - A * exp(-n/lambda)
to post-session-boundary performance from zero-based trials 0 through 299.
P_inf is estimated from the post-boundary trajectory and is not forced to the
empirical pre-boundary baseline. A is the fitted recovery amplitude from P(0)
to P_inf, and lambda is the recovery timescale in trials.

Raw rewarded/correct trials are summarized in non-overlapping 20-trial bins.
The fitted value for a bin is the mean model prediction across every integer
trial in that bin. Fits use unweighted bounded least squares, deterministic
multiple starts, P_inf and A bounds [0,1], lambda bounds [1,300], and at least
8 finite bins. Predictions are not shifted, normalized, or clipped.

The primary Protocol x Genotype group trajectory is boundary-weighted: every
valid boundary-level bin is an observation, so animals with more boundaries
contribute more rows. Uncertainty uses {bootstrap_replicates} animal-cluster
bootstrap replicates with seed {bootstrap_seed}. Animals are resampled with
replacement and every selected animal brings all of its boundaries. Repeated
animal draws retain cluster multiplicity. This preserves the boundary-weighted
estimand and does not switch to animal-equal weighting. Reported intervals are
animal-cluster bootstrap 95% CIs.

Empirical PreBoundaryBaseline uses up to 300 trials from the previous session.
InitialPostPerformance uses the first 50 available post trials. DropMagnitude
and AbsoluteDrop are the same absolute proportion difference. PercentDrop is
calculated boundary-by-boundary as 100 * AbsoluteDrop / PreBoundaryBaseline
before aggregation; baselines with absolute value <= {PERCENT_BASELINE_ATOL:g}
are flagged invalid and not divided.

PInfWarning, AWarning, and LambdaWarning identify estimates within 1% of their
parameter bounds. P0Warning identifies fitted P(0) outside [0,1].
BoundaryWarning is the composite of those flags and also marks failed fits.
Failed, non-finite, insufficient-bin, and parameter-warning fits must not be
treated as valid biological parameter estimates.

Identifiability caveats:
- P_inf, A, and lambda trade off if performance has not plateaued by trial 299.
- Flat/noisy trajectories can put A near zero and leave lambda unidentified.
- Each fit has at most 15 binned observations.
- A partly extrapolates to trial zero because the first observation averages
  trials 0-19.
- Independent P_inf and A bounds can allow P(0) below zero; this is flagged and
  never clipped or shifted.
- Fitted A measures recovery toward the fitted post-session asymptote, whereas
  empirical DropMagnitude compares the previous-session baseline with the first
  50 post trials. Directional agreement does not require numerical equality.
- Bootstrap intervals can be skewed or truncated by parameter bounds.

This is a descriptive/modeling analysis. Animal-cluster bootstrap 95% CIs
describe uncertainty in each group curve; they are not inferential WT/HET
genotype-comparison tests. No statistical significance claim should be made
without an explicit inferential analysis.
"""
    (output_dir / "README_session_boundary_recovery_fit.txt").write_text(
        text, encoding="utf-8"
    )


def write_outputs(
    group_summary,
    animal_summary,
    group_curve,
    output_dir,
    bootstrap_replicates,
    bootstrap_seed,
):
    """Write all requested tables, static figures, and explanatory README."""

    output_dir.mkdir(parents=True, exist_ok=True)
    group_summary.to_csv(output_dir / "recovery_group_fit_summary.csv", index=False)
    animal_summary.to_csv(
        output_dir / "recovery_animal_fit_summary.csv", index=False
    )
    group_curve.to_csv(
        output_dir / "recovery_group_aligned_curve.csv", index=False
    )
    for protocol in ("AB", "CD"):
        plot_protocol_recovery(
            group_summary,
            group_curve,
            protocol,
            output_dir / f"{protocol}_exponential_recovery_fit.png",
        )
    plot_parameter_comparison(
        animal_summary, output_dir / "recovery_parameter_comparison.png"
    )
    plot_fitted_vs_empirical_drop(
        animal_summary, output_dir / "fitted_vs_empirical_drop.png"
    )
    write_readme(output_dir, bootstrap_replicates, bootstrap_seed)


def run_synthetic_check():
    """Fit a seeded noisy trajectory with known recovery parameters."""

    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    starts = np.arange(0, POST_BOUNDARY_TRIALS, BIN_TRIALS)
    ends = starts + BIN_TRIALS - 1
    expected = binned_model_prediction(starts, ends, np.array([0.8, 0.2, 75.0]))
    curve = pd.DataFrame(
        {
            "BinStart": starts,
            "BinEnd": ends,
            "ObservedMean": expected + rng.normal(0, 0.004, len(starts)),
        }
    )
    return fit_exponential_recovery(curve)


def analyze_dataset(model, bootstrap_replicates, bootstrap_seed):
    """Extract all AB/CD boundary data and produce fitted summaries."""

    bin_rows = []
    metric_rows = []
    animal_info = model.data_index.drop_duplicates("Animal").set_index("Animal")
    for animal in model.data_index["Animal"].unique():
        genotype = animal_info.loc[animal, "Genotype"]
        gender = animal_info.loc[animal, "Gender"]
        identity = {"Animal": animal, "Genotype": genotype, "Gender": gender}
        for protocol in ("AB", "CD"):
            sessions = load_protocol_sessions(model, animal, protocol)
            bins, metrics = extract_boundary_data(sessions)
            for row in bins.to_dict("records"):
                bin_rows.append(identity | {"Protocol": protocol} | row)
            for row in metrics.to_dict("records"):
                metric_rows.append(identity | {"Protocol": protocol} | row)
    boundary_bins = pd.DataFrame(bin_rows)
    boundary_metrics = pd.DataFrame(metric_rows)
    animal_summary = fit_animal_curves(boundary_bins, boundary_metrics)
    group_summary, group_curve = fit_group_curves(
        boundary_bins,
        boundary_metrics,
        n_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return group_summary, animal_summary, group_curve, boundary_bins, boundary_metrics


def expected_output_paths(output_dir):
    return [
        output_dir / "recovery_group_fit_summary.csv",
        output_dir / "recovery_animal_fit_summary.csv",
        output_dir / "recovery_group_aligned_curve.csv",
        output_dir / "AB_exponential_recovery_fit.png",
        output_dir / "CD_exponential_recovery_fit.png",
        output_dir / "recovery_parameter_comparison.png",
        output_dir / "fitted_vs_empirical_drop.png",
        output_dir / "README_session_boundary_recovery_fit.txt",
    ]


def guard_against_overwrite(output_dir):
    existing = [path for path in expected_output_paths(output_dir) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing recovery-fit outputs:\n"
            + "\n".join(str(path) for path in existing)
        )


def format_group_table(group_summary):
    lines = [
        "Protocol | Genotype | NAnimals | NBoundaries | P_inf "
        "[animal-cluster bootstrap 95% CI] | A [animal-cluster bootstrap 95% CI] "
        "| lambda [animal-cluster bootstrap 95% CI] | RMSE | R2"
    ]
    for protocol, genotype in GROUP_ORDER:
        selected = group_summary[
            (group_summary["Protocol"] == protocol)
            & (group_summary["Genotype"].astype(str).str.upper() == genotype)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0]
        lines.append(
            f"{protocol} | {genotype} | {int(row['NAnimals'])} | "
            f"{int(row['NBoundaries'])} | {row['P_inf']:.3f} "
            f"[{row['P_inf_CI_low']:.3f}, {row['P_inf_CI_high']:.3f}] | "
            f"{row['A']:.3f} [{row['A_CI_low']:.3f}, {row['A_CI_high']:.3f}] | "
            f"{row['lambda']:.1f} "
            f"[{row['lambda_CI_low']:.1f}, {row['lambda_CI_high']:.1f}] | "
            f"{row['RMSE']:.4f} | {row['R2']:.3f}"
        )
    return lines


def qualitative_findings(group_summary):
    lookup = group_summary.set_index(["Protocol", "Genotype"])
    comparisons = []
    for genotype in ("WT", "HET"):
        if ("AB", genotype) in lookup.index and ("CD", genotype) in lookup.index:
            comparisons.append(
                f"CD {genotype} A {'>' if lookup.at[('CD', genotype), 'A'] > lookup.at[('AB', genotype), 'A'] else '<='} AB {genotype} A"
            )
    valid_group = group_summary[
        group_summary["FitSuccess"].astype(bool)
        & ~group_summary["BoundaryWarning"].astype(bool)
    ]
    largest_a = (
        valid_group.loc[valid_group["A"].idxmax(), ["Protocol", "Genotype"]]
        if not valid_group.empty
        else None
    )
    largest_lambda = (
        valid_group.loc[
            valid_group["lambda"].idxmax(), ["Protocol", "Genotype"]
        ]
        if not valid_group.empty
        else None
    )
    all_directional = bool(
        (
            np.sign(group_summary["A"])
            == np.sign(group_summary["MeanAbsoluteDrop"])
        ).all()
        and (
            np.sign(group_summary["A"])
            == np.sign(group_summary["MeanPercentDrop"])
        ).all()
    )
    return (
        "; ".join(comparisons)
        + (f"; largest valid A={largest_a['Protocol']} {largest_a['Genotype']}" if largest_a is not None else "; no valid non-warning A")
        + (f"; longest valid lambda={largest_lambda['Protocol']} {largest_lambda['Genotype']}" if largest_lambda is not None else "; no valid non-warning lambda")
        + f"; fitted A agrees in sign with empirical drops={all_directional}. "
        "Descriptive comparisons only; no genotype-effect inference."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--strain")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--synthetic-test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.synthetic_test_only:
        fit = run_synthetic_check()
        print(
            "Synthetic recovery fit: "
            f"P_inf={fit['P_inf']:.4f} (target 0.80), "
            f"A={fit['A']:.4f} (target 0.20), "
            f"lambda={fit['lambda']:.2f} (target 75), "
            f"success={fit['FitSuccess']}"
        )
        return
    if not args.root_dir or not args.output_dir:
        raise SystemExit("--root-dir and --output-dir are required")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be at least 1")

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    guard_against_overwrite(output_dir)
    strain = args.strain or root_dir.name
    model = ReadOnlyBehDataOdor(str(root_dir), strain)
    group_summary, animal_summary, group_curve, _, _ = analyze_dataset(
        model,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_outputs(
        group_summary,
        animal_summary,
        group_curve,
        output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print("Session-boundary exponential recovery analysis complete: " + str(output_dir))
    print("\n".join(format_group_table(group_summary)))
    print("Qualitative findings: " + qualitative_findings(group_summary))


if __name__ == "__main__":
    main()
