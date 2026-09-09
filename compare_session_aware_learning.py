"""Compare continuous and session-aware sigmoid learning models.

This is an external, read-only diagnostic. It imports session-loading and
learning-curve helpers from ``analyze_session_boundary`` but never invokes
that module's ``main`` function or writes through ``BehDataOdor``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from analyze_session_boundary import (
    ReadOnlyBehDataOdor,
    concatenate_with_boundaries,
    load_protocol_sessions,
    rewarded_vector,
    smooth_learning_curve,
)


RUNNING_WINDOW = 60
SMOOTH_WINDOW = 500
MIN_FIT_POINTS = 8
DEFAULT_N_STARTS = 24
DEFAULT_SEED = 20260826
PARAMETER_EPSILON = 1e-8

OUTPUT_COLUMNS = [
    "Animal",
    "Genotype",
    "Gender",
    "Protocol",
    "NTrials",
    "NFitPoints",
    "NSessions",
    "SessionNumbers",
    "SessionDates",
    "p_low_cont",
    "p_high_cont",
    "k_cont",
    "tau_cont",
    "RSS_cont",
    "RMSE_cont",
    "R2_cont",
    "AIC_cont",
    "BIC_cont",
    "success_cont",
    "successful_starts_cont",
    "message_cont",
    "p_low_session",
    "p_high_session",
    "k_session",
    "tau_session",
    "reset_A",
    "recovery_lambda",
    "RSS_session",
    "RMSE_session",
    "R2_session",
    "AIC_session",
    "BIC_session",
    "success_session",
    "successful_starts_session",
    "message_session",
    "DeltaRMSE",
    "DeltaAIC",
    "DeltaBIC",
    "DeltaTau",
    "DeltaK",
    "FailureReason",
]


def continuous_probability(x, p_low, p_high, k, tau):
    """Continuous latent learning probability."""

    x = np.asarray(x, dtype=float)
    return p_low + (p_high - p_low) * expit(k * (x - tau))


def session_aware_probability(
    x,
    session_index,
    trial_in_session,
    p_low,
    p_high,
    k,
    tau,
    reset_A,
    recovery_lambda,
):
    """Latent learning probability with a shared post-session reset."""

    base = continuous_probability(x, p_low, p_high, k, tau)
    session_index = np.asarray(session_index)
    trial_in_session = np.asarray(trial_in_session, dtype=float)
    reset = reset_A * np.exp(-trial_in_session / recovery_lambda)
    return base - reset * (session_index > 0)


def observed_scale_prediction(latent_probability):
    """Apply the pipeline's exact 60-trial then centered-500 operator."""

    return smooth_learning_curve(
        np.asarray(latent_probability, dtype=float),
        running_window=RUNNING_WINDOW,
        smooth_window=SMOOTH_WINDOW,
    )


def _decode_sigmoid(theta):
    p_low, span_fraction, k, tau = theta[:4]
    p_high = p_low + (1.0 - p_low) * span_fraction
    return float(p_low), float(p_high), float(k), float(tau)


def _encode_sigmoid(p_low, p_high, k, tau):
    p_low = float(np.clip(p_low, 0.0, 1.0 - PARAMETER_EPSILON))
    p_high = float(np.clip(p_high, p_low + PARAMETER_EPSILON, 1.0))
    denominator = max(1.0 - p_low, PARAMETER_EPSILON)
    span_fraction = np.clip(
        (p_high - p_low) / denominator,
        PARAMETER_EPSILON,
        1.0,
    )
    return np.array([p_low, span_fraction, k, tau], dtype=float)


def _model_prediction(
    theta,
    x,
    session_index,
    trial_in_session,
    session_aware,
):
    p_low, p_high, k, tau = _decode_sigmoid(theta)

    if session_aware:
        latent = session_aware_probability(
            x,
            session_index,
            trial_in_session,
            p_low,
            p_high,
            k,
            tau,
            float(theta[4]),
            float(theta[5]),
        )
    else:
        latent = continuous_probability(x, p_low, p_high, k, tau)

    return latent, observed_scale_prediction(latent)


def _initial_sigmoid_guesses(
    y_valid,
    n_trials,
    n_starts,
    rng,
):
    q10, q90 = np.nanpercentile(y_valid, [10, 90])
    p_low = float(np.clip(q10, 0.01, 0.9))
    p_high = float(np.clip(max(q90, p_low + 0.05), p_low + 0.01, 0.99))
    base = _encode_sigmoid(
        p_low,
        p_high,
        0.005,
        (n_trials + 1.0) / 2.0,
    )
    guesses = [base]

    for _ in range(max(0, n_starts - 1)):
        random_low = rng.uniform(0.0, min(0.8, p_high))
        random_high = rng.uniform(random_low + 0.01, 1.0)
        random_k = np.exp(rng.uniform(np.log(1e-5), np.log(0.2)))
        random_tau = rng.uniform(1.0, float(n_trials))
        guesses.append(
            _encode_sigmoid(
                random_low,
                random_high,
                random_k,
                random_tau,
            )
        )

    return guesses


def _fit_model(
    observed,
    valid,
    x,
    session_index,
    trial_in_session,
    max_session_length,
    n_starts,
    seed,
    session_aware,
    continuous_theta=None,
):
    rng = np.random.default_rng(seed)
    guesses = _initial_sigmoid_guesses(
        observed[valid],
        len(observed),
        n_starts,
        rng,
    )
    sigmoid_bounds = [
        (0.0, 1.0 - PARAMETER_EPSILON),
        (PARAMETER_EPSILON, 1.0),
        (1e-8, 1.0),
        (1.0, float(len(observed))),
    ]

    if session_aware:
        session_guesses = []
        if continuous_theta is not None:
            session_guesses.append(
                np.concatenate(
                    [
                        np.asarray(continuous_theta[:4], dtype=float),
                        [0.02, min(100.0, float(max_session_length))],
                    ]
                )
            )

        for guess in guesses:
            reset_A = rng.uniform(0.03, 0.6)
            recovery_lambda = np.exp(
                rng.uniform(0.0, np.log(float(max_session_length)))
            )
            session_guesses.append(
                np.concatenate([guess, [reset_A, recovery_lambda]])
            )

        guesses = session_guesses[:n_starts]
        bounds = sigmoid_bounds + [
            (0.0, 1.0),
            (1.0, float(max_session_length)),
        ]
    else:
        bounds = sigmoid_bounds

    def objective(theta):
        _, predicted = _model_prediction(
            theta,
            x,
            session_index,
            trial_in_session,
            session_aware,
        )
        residual = observed[valid] - predicted[valid]
        if not np.all(np.isfinite(residual)):
            return np.finfo(float).max / 100.0
        return float(np.dot(residual, residual))

    results = []
    for guess in guesses:
        result = minimize(
            objective,
            guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol": 1e-12,
                "maxls": 50,
            },
        )
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            results.append(result)

    successful = [result for result in results if result.success]
    candidates = successful if successful else results

    if not candidates:
        return {
            "success": False,
            "message": "No initialization returned a finite objective",
            "successful_starts": 0,
            "theta": None,
            "latent": None,
            "prediction": None,
        }

    best = min(candidates, key=lambda result: result.fun)
    latent, prediction = _model_prediction(
        best.x,
        x,
        session_index,
        trial_in_session,
        session_aware,
    )
    messages = sorted({str(result.message) for result in results})

    return {
        "success": bool(best.success),
        "message": str(best.message) if best.success else "; ".join(messages),
        "successful_starts": len(successful),
        "theta": np.asarray(best.x, dtype=float),
        "latent": latent,
        "prediction": prediction,
    }


def _fit_metrics(observed, prediction, valid, fitted_parameter_count):
    residual = observed[valid] - prediction[valid]
    rss = float(np.dot(residual, residual))
    n = int(np.sum(valid))
    rmse = float(np.sqrt(rss / n))
    centered = observed[valid] - np.mean(observed[valid])
    tss = float(np.dot(centered, centered))
    r_squared = float(1.0 - rss / tss) if tss > 0 else np.nan

    variance_mle = max(rss / n, np.finfo(float).tiny)
    log_likelihood = -0.5 * n * (
        np.log(2.0 * np.pi) + 1.0 + np.log(variance_mle)
    )
    aic = float(2 * fitted_parameter_count - 2 * log_likelihood)
    bic = float(
        fitted_parameter_count * np.log(n) - 2 * log_likelihood
    )

    return {
        "RSS": rss,
        "RMSE": rmse,
        "R2": r_squared,
        "AIC": aic,
        "BIC": bic,
    }


def _blank_fit_result(n_trials, n_fit_points, n_sessions, reason):
    result = {column: np.nan for column in OUTPUT_COLUMNS}
    result.update(
        {
            "NTrials": int(n_trials),
            "NFitPoints": int(n_fit_points),
            "NSessions": int(n_sessions),
            "success_cont": False,
            "success_session": False,
            "successful_starts_cont": 0,
            "successful_starts_session": 0,
            "message_cont": "Not fit",
            "message_session": "Not fit",
            "FailureReason": reason,
            "_prediction_cont": None,
            "_prediction_session": None,
            "_latent_cont": None,
            "_latent_session": None,
        }
    )
    return result


def fit_model_comparison(
    observed_curve,
    session_index,
    trial_in_session,
    n_starts=DEFAULT_N_STARTS,
    seed=DEFAULT_SEED,
):
    """Fit both models to a common observed smoothed learning curve."""

    observed = np.asarray(observed_curve, dtype=float)
    session_index = np.asarray(session_index, dtype=int)
    trial_in_session = np.asarray(trial_in_session, dtype=int)

    if not (
        len(observed) == len(session_index) == len(trial_in_session)
    ):
        raise ValueError(
            "observed_curve, session_index, and trial_in_session must "
            "have identical lengths"
        )

    n_trials = len(observed)
    valid = np.isfinite(observed)
    n_fit_points = int(np.sum(valid))
    n_sessions = int(np.unique(session_index).size) if n_trials else 0
    result = _blank_fit_result(
        n_trials,
        n_fit_points,
        n_sessions,
        "",
    )

    if n_trials == 0:
        result["FailureReason"] = "No valid trials"
        return result

    if n_fit_points < MIN_FIT_POINTS:
        result["FailureReason"] = (
            f"Insufficient finite fit points: {n_fit_points} < {MIN_FIT_POINTS}"
        )
        return result

    x = np.arange(1, n_trials + 1, dtype=float)
    session_lengths = np.bincount(session_index)
    max_session_length = int(np.max(session_lengths))

    continuous = _fit_model(
        observed,
        valid,
        x,
        session_index,
        trial_in_session,
        max_session_length,
        n_starts,
        seed,
        session_aware=False,
    )
    result["success_cont"] = continuous["success"]
    result["successful_starts_cont"] = continuous["successful_starts"]
    result["message_cont"] = continuous["message"]

    failure_reasons = []
    if continuous["theta"] is not None:
        p_low, p_high, k, tau = _decode_sigmoid(continuous["theta"])
        metrics = _fit_metrics(
            observed,
            continuous["prediction"],
            valid,
            fitted_parameter_count=5,
        )
        result.update(
            {
                "p_low_cont": p_low,
                "p_high_cont": p_high,
                "k_cont": k,
                "tau_cont": tau,
                "RSS_cont": metrics["RSS"],
                "RMSE_cont": metrics["RMSE"],
                "R2_cont": metrics["R2"],
                "AIC_cont": metrics["AIC"],
                "BIC_cont": metrics["BIC"],
                "_prediction_cont": continuous["prediction"],
                "_latent_cont": continuous["latent"],
            }
        )
    if not continuous["success"]:
        failure_reasons.append(
            f"Continuous optimization failed: {continuous['message']}"
        )

    if n_sessions < 2:
        result["message_session"] = (
            "Session-aware model requires at least 2 nonempty sessions"
        )
        failure_reasons.append(result["message_session"])
        result["FailureReason"] = "; ".join(failure_reasons)
        return result

    session_fit = _fit_model(
        observed,
        valid,
        x,
        session_index,
        trial_in_session,
        max_session_length,
        n_starts,
        seed + 1,
        session_aware=True,
        continuous_theta=continuous["theta"],
    )
    result["success_session"] = session_fit["success"]
    result["successful_starts_session"] = session_fit[
        "successful_starts"
    ]
    result["message_session"] = session_fit["message"]

    if session_fit["theta"] is not None:
        p_low, p_high, k, tau = _decode_sigmoid(session_fit["theta"])
        metrics = _fit_metrics(
            observed,
            session_fit["prediction"],
            valid,
            fitted_parameter_count=7,
        )
        result.update(
            {
                "p_low_session": p_low,
                "p_high_session": p_high,
                "k_session": k,
                "tau_session": tau,
                "reset_A": float(session_fit["theta"][4]),
                "recovery_lambda": float(session_fit["theta"][5]),
                "RSS_session": metrics["RSS"],
                "RMSE_session": metrics["RMSE"],
                "R2_session": metrics["R2"],
                "AIC_session": metrics["AIC"],
                "BIC_session": metrics["BIC"],
                "_prediction_session": session_fit["prediction"],
                "_latent_session": session_fit["latent"],
            }
        )
    if not session_fit["success"]:
        failure_reasons.append(
            f"Session-aware optimization failed: {session_fit['message']}"
        )

    comparison_pairs = {
        "DeltaRMSE": ("RMSE_session", "RMSE_cont"),
        "DeltaAIC": ("AIC_session", "AIC_cont"),
        "DeltaBIC": ("BIC_session", "BIC_cont"),
        "DeltaTau": ("tau_session", "tau_cont"),
        "DeltaK": ("k_session", "k_cont"),
    }
    for output, (session_key, continuous_key) in comparison_pairs.items():
        if np.isfinite(result[session_key]) and np.isfinite(
            result[continuous_key]
        ):
            result[output] = result[session_key] - result[continuous_key]

    result["FailureReason"] = "; ".join(failure_reasons)
    return result


def prepare_protocol_data(session_frames):
    """Build observed and session-aware coordinates without losing identity."""

    concatenated, boundaries = concatenate_with_boundaries(session_frames)
    if concatenated.empty:
        return {
            "concatenated": concatenated,
            "boundaries": boundaries,
            "observed": np.array([], dtype=float),
            "session_index": np.array([], dtype=int),
            "trial_in_session": np.array([], dtype=int),
        }

    rewarded = rewarded_vector(concatenated)
    observed = smooth_learning_curve(
        rewarded,
        running_window=RUNNING_WINDOW,
        smooth_window=SMOOTH_WINDOW,
    )
    session_index = np.concatenate(
        [np.full(len(frame), index, dtype=int) for index, frame in enumerate(session_frames)]
    )
    trial_in_session = np.concatenate(
        [np.arange(len(frame), dtype=int) for frame in session_frames]
    )

    return {
        "concatenated": concatenated,
        "boundaries": boundaries,
        "observed": observed,
        "session_index": session_index,
        "trial_in_session": trial_in_session,
    }


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def plot_animal_comparison(
    row,
    observed,
    boundaries,
    output_dir,
):
    animal = row["Animal"]
    protocol = row["Protocol"]
    genotype = row["Genotype"]
    x = np.arange(1, len(observed) + 1)
    valid = np.isfinite(observed)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    if np.any(valid):
        ax.plot(
            x[valid],
            observed[valid],
            color="black",
            linewidth=2,
            label="Observed smoothed curve",
        )

    plotted_values = [observed[valid]] if np.any(valid) else []
    for key, label, color in (
        ("_prediction_cont", "Continuous sigmoid", "tab:blue"),
        ("_prediction_session", "Session-aware sigmoid", "tab:orange"),
    ):
        prediction = row.get(key)
        if prediction is None:
            continue
        prediction = np.asarray(prediction, dtype=float)
        prediction_valid = np.isfinite(prediction)
        ax.plot(
            x[prediction_valid],
            prediction[prediction_valid],
            color=color,
            linewidth=2,
            label=label,
        )
        plotted_values.append(prediction[prediction_valid])

    for number, boundary in enumerate(boundaries):
        ax.axvline(
            boundary + 0.5,
            color="0.5",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Session boundary" if number == 0 else None,
        )

    statistics = (
        f"Continuous: RMSE={_fmt(row['RMSE_cont'])}, "
        f"AIC={_fmt(row['AIC_cont'], 1)}, BIC={_fmt(row['BIC_cont'], 1)}\n"
        f"Session-aware: RMSE={_fmt(row['RMSE_session'])}, "
        f"AIC={_fmt(row['AIC_session'], 1)}, BIC={_fmt(row['BIC_session'], 1)}\n"
        f"reset_A={_fmt(row['reset_A'])}, "
        f"recovery_lambda={_fmt(row['recovery_lambda'], 1)}"
    )
    ax.text(
        0.02,
        0.98,
        statistics,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    if row["FailureReason"]:
        ax.text(
            0.02,
            0.02,
            str(row["FailureReason"]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="tab:red",
        )

    finite_values = [values for values in plotted_values if len(values)]
    if finite_values:
        values = np.concatenate(finite_values)
        lower = min(0.0, float(np.nanmin(values)))
        upper = max(1.0, float(np.nanmax(values)))
        padding = max(0.05, 0.05 * (upper - lower))
        ax.set_ylim(lower - padding, upper + padding)
    else:
        ax.set_ylim(0, 1)
        ax.text(
            0.5,
            0.5,
            "No valid smoothed observations",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Concatenated trial")
    ax.set_ylabel("Smoothed P(correct)")
    ax.set_title(f"{animal} ({genotype}) {protocol}: model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    animal_dir = output_dir / str(animal)
    animal_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        animal_dir / f"{animal}_{protocol}_model_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def _genotype_order(frame):
    values = [str(value) for value in frame["Genotype"].dropna().unique()]
    preferred = [value for value in ("WT", "HET") if value in values]
    return preferred + sorted(value for value in values if value not in preferred)


def _plot_delta_by_genotype(ax, frame, metric):
    genotypes = _genotype_order(frame)
    plotted = False
    for position, genotype in enumerate(genotypes):
        values = frame.loc[frame["Genotype"].astype(str) == genotype, metric].dropna()
        if values.empty:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(
            position + np.asarray(jitter),
            values,
            alpha=0.8,
            color="tab:blue" if genotype == "WT" else "tab:orange",
        )
        median = float(values.median())
        ax.plot([position - 0.18, position + 0.18], [median, median], color="black", linewidth=2)
        plotted = True

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(genotypes)), genotypes)
    ax.set_title(metric)
    ax.set_ylabel("Session-aware minus continuous")
    ax.spines[["top", "right"]].set_visible(False)
    if not plotted:
        ax.text(0.5, 0.5, "No valid comparisons", transform=ax.transAxes, ha="center")


def _plot_parameter_pair(ax, frame, continuous_key, session_key, label, log_scale=False):
    colors = {"WT": "tab:blue", "HET": "tab:orange"}
    plotted = False
    for genotype in _genotype_order(frame):
        valid = frame[[continuous_key, session_key]].notna().all(axis=1)
        subset = frame[valid & (frame["Genotype"].astype(str) == genotype)]
        if subset.empty:
            continue
        ax.scatter(
            subset[continuous_key],
            subset[session_key],
            label=genotype,
            alpha=0.8,
            color=colors.get(genotype),
        )
        plotted = True

    if plotted:
        paired = frame[[continuous_key, session_key]].dropna().to_numpy(dtype=float)
        lower = float(np.min(paired))
        upper = float(np.max(paired))
        if log_scale:
            positive = paired[paired > 0]
            lower = float(np.min(positive))
            upper = float(np.max(positive))
            ax.set_xscale("log")
            ax.set_yscale("log")
        if np.isclose(lower, upper):
            lower *= 0.9
            upper *= 1.1
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No valid comparisons", transform=ax.transAxes, ha="center")

    ax.set_xlabel(f"Continuous {label}")
    ax.set_ylabel(f"Session-aware {label}")
    ax.set_title(f"{label}: continuous vs session-aware")
    ax.spines[["top", "right"]].set_visible(False)


def plot_group_comparison(comparison, protocol, output_dir):
    frame = comparison[comparison["Protocol"] == protocol].copy()
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, start : start + 2]) for start in (0, 2, 4)]
    for ax, metric in zip(axes, ("DeltaAIC", "DeltaBIC", "DeltaRMSE")):
        _plot_delta_by_genotype(ax, frame, metric)

    tau_ax = fig.add_subplot(grid[1, :3])
    k_ax = fig.add_subplot(grid[1, 3:])
    _plot_parameter_pair(tau_ax, frame, "tau_cont", "tau_session", "tau")
    _plot_parameter_pair(k_ax, frame, "k_cont", "k_session", "k", log_scale=True)
    fig.suptitle(f"{protocol}: continuous vs session-aware learning models")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{protocol}_model_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_group_summary(comparison):
    columns = [
        "Protocol",
        "Genotype",
        "NAnimals",
        "NBothSuccessful",
        "MeanDeltaRMSE",
        "MedianDeltaRMSE",
        "SEMDeltaRMSE",
        "MeanDeltaAIC",
        "MedianDeltaAIC",
        "SEMDeltaAIC",
        "MeanDeltaBIC",
        "MedianDeltaBIC",
        "SEMDeltaBIC",
        "FractionSessionAwareLowerRMSE",
        "FractionSessionAwareLowerAIC",
        "FractionSessionAwareLowerBIC",
        "MeanDeltaTau",
        "MeanDeltaK",
    ]
    rows = []
    for (protocol, genotype), frame in comparison.groupby(
        ["Protocol", "Genotype"],
        dropna=False,
    ):
        row = {
            "Protocol": protocol,
            "Genotype": genotype,
            "NAnimals": len(frame),
            "NBothSuccessful": int((frame["success_cont"] & frame["success_session"]).sum()),
        }
        for metric in ("DeltaRMSE", "DeltaAIC", "DeltaBIC"):
            values = frame[metric].dropna()
            suffix = metric.replace("Delta", "")
            row[f"MeanDelta{suffix}"] = values.mean()
            row[f"MedianDelta{suffix}"] = values.median()
            row[f"SEMDelta{suffix}"] = values.sem()
            row[f"FractionSessionAwareLower{suffix}"] = (
                float((values < 0).mean()) if len(values) else np.nan
            )
        row["MeanDeltaTau"] = frame["DeltaTau"].mean()
        row["MeanDeltaK"] = frame["DeltaK"].mean()
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _readme_text(n_starts):
    return f"""Session-aware sigmoid model comparison

Purpose
This is an external diagnostic/model-comparison analysis. It reads existing
processed behavioral CSVs through ReadOnlyBehDataOdor and never invokes
analyze_session_boundary.main, modifies source CSVs, or reuses saved sigmoid
parameters.

Data inclusion and observation curve
AB includes sessions before the first protocol containing AB-CD. CD includes
AB-CD sessions with ProtocolDay <= 3 and retains only schedule > 2 trials.
Miss trials are excluded. Rewards 2 and 3 are mapped to rewarded/correct, then
converted to a binary vector. The observed curve uses the exact pipeline
operator: a 60-trial running reward rate followed by a 500-trial centered
rolling mean with min_periods=1.

Models and fitting scale
The continuous latent model is p_low + (p_high-p_low)/(1+exp(-k*(t-tau))).
The session-aware latent model subtracts A*exp(-n_session/lambda) in every
session after the first, using zero-based within-session trial. Critically,
each complete latent model trajectory is passed through the same 60/500
operator before residuals are compared with the observed smoothed curve.
Predictions are not clipped for fitting or plotting, so pathological reset
estimates remain visible.

Bounds and optimization
Both models minimize RSS over the identical finite observed points with
L-BFGS-B and {n_starts} seeded initializations. Bounds are 0 <= p_low <
p_high <= 1, 1e-8 <= k <= 1, tau within the concatenated trial range,
0 <= A <= 1, and 1 <= lambda <= the longest included session. The sigmoid
ordering is enforced by p_high=p_low+(1-p_low)*span_fraction.

Metrics and working likelihood
RMSE and R2 derive from RSS over the common fit points. AIC and BIC use an
independent, homoscedastic Gaussian residual working likelihood with variance
estimated as RSS/n: logL=-n/2*[log(2*pi)+1+log(RSS/n)]. Continuous AIC/BIC
count five parameters (four curve parameters plus residual variance), while
session-aware criteria count seven (six model parameters plus variance).
Because the 60/500 smoothing induces residual autocorrelation, AIC/BIC are
comparative diagnostics under this working likelihood, not a trial-level
Bernoulli likelihood.

Interpretation
Delta metrics are session-aware minus continuous. Negative DeltaRMSE,
DeltaAIC, or DeltaBIC favors the session-aware model. DeltaTau and DeltaK
show whether explicit resets alter estimated rising-phase parameters.
FailureReason and optimizer messages identify missing sessions, insufficient
finite points, and failed or partial fits. Finite estimates are retained even
when an optimizer reports failure.

Synthetic validation limitation
The direct noisy-latent fixture tests parameter recovery after applying the
matched observation operator. The separate fixed-seed Bernoulli fixture tests
end-to-end comparative fit. Even with matched smoothing of predictions,
smoothing attenuates high-frequency information and broadens resets, so a
single Bernoulli realization is not required to recover its exact generating
A or lambda. Its deliberately strong reset may improve AIC/BIC, but that is a
validation-fixture result rather than a mathematical guarantee for every
realization or real dataset.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strain")
    parser.add_argument("--n-starts", type=int, default=DEFAULT_N_STARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strain = args.strain or root_dir.name

    model = ReadOnlyBehDataOdor(str(root_dir), strain)
    animal_info = model.data_index.drop_duplicates("Animal").set_index("Animal")
    rows = []

    for animal in model.data_index["Animal"].unique():
        genotype = animal_info.loc[animal, "Genotype"]
        gender = animal_info.loc[animal, "Gender"]

        for protocol in ("AB", "CD"):
            session_frames = load_protocol_sessions(model, animal, protocol)
            prepared = prepare_protocol_data(session_frames)
            fit = fit_model_comparison(
                prepared["observed"],
                prepared["session_index"],
                prepared["trial_in_session"],
                n_starts=args.n_starts,
                seed=args.seed,
            )
            fit.update(
                {
                    "Animal": animal,
                    "Genotype": genotype,
                    "Gender": gender,
                    "Protocol": protocol,
                    "SessionNumbers": ";".join(
                        str(int(frame["_session_number"].iloc[0]))
                        for frame in session_frames
                    ),
                    "SessionDates": ";".join(
                        str(frame["_date"].iloc[0]) for frame in session_frames
                    ),
                }
            )
            if not session_frames and not fit["FailureReason"]:
                fit["FailureReason"] = (
                    f"No valid {protocol} sessions under inclusion rules"
                )
            rows.append(fit)
            plot_animal_comparison(
                fit,
                prepared["observed"],
                prepared["boundaries"],
                output_dir,
            )

    comparison = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    comparison.to_csv(
        output_dir / "session_aware_model_comparison.csv",
        index=False,
    )
    group_summary = build_group_summary(comparison)
    group_summary.to_csv(
        output_dir / "session_aware_model_group_summary.csv",
        index=False,
    )

    for protocol in ("AB", "CD"):
        plot_group_comparison(comparison, protocol, output_dir)

    (output_dir / "README_session_aware_model_comparison.txt").write_text(
        _readme_text(args.n_starts),
        encoding="utf-8",
    )

    print(f"Session-aware model comparison complete: {output_dir}")
    print(f"Animal-protocol rows: {len(comparison)}")
    print(
        "Both models successful: "
        f"{int((comparison['success_cont'] & comparison['success_session']).sum())}"
    )


if __name__ == "__main__":
    main()
