"""Test session-boundary reset/recovery sensitivity to alignment windows.

This is a standalone, read-only analysis wrapper. It reuses the behavioral
loading, inclusion, and reward definitions from ``analyze_session_boundary``
and never writes to the source-data tree or the reference 300/500 output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_session_boundary import (
    BOUNDARY_BIN_TRIALS,
    RECOVERY_WINDOW_TRIALS,
    ReadOnlyBehDataOdor,
    load_protocol_sessions,
    rewarded_vector,
)


WINDOW_CONFIGS = ((200, 200), (300, 300))
REFERENCE_WINDOW = (300, 500)
PERCENT_BASELINE_ATOL = 1e-12
SUMMARY_STATISTICS = (
    "NBoundaries",
    "MeanDrop",
    "MedianDrop",
    "SDDrop",
    "FractionRecovered",
    "MedianRecoveryTrial",
    "MeanRecoveryTrial",
    "FractionPositiveDrop",
    "NPercentDropValid",
    "MeanPercentDrop",
    "MedianPercentDrop",
    "SDPercentDrop",
    "Q1PercentDrop",
    "Q3PercentDrop",
    "MeanPreBoundaryBaseline",
    "MedianPreBoundaryBaseline",
    "MeanInitialPostPerformance",
    "MedianInitialPostPerformance",
)


def window_label(pre_trials, post_trials):
    return f"pre{pre_trials}/post{post_trials}"


def drop_metrics(baseline, initial_post_performance):
    """Calculate absolute and relative drops for one boundary."""

    absolute_drop = baseline - initial_post_performance
    percent_valid = bool(
        np.isfinite(baseline)
        and np.isfinite(initial_post_performance)
        and not np.isclose(baseline, 0.0, rtol=0.0, atol=PERCENT_BASELINE_ATOL)
    )
    percent_drop = (
        100.0 * absolute_drop / baseline if percent_valid else np.nan
    )
    return absolute_drop, percent_drop, percent_valid


def boundary_aligned_analysis(
    session_frames,
    pre_trials,
    post_trials,
    bin_trials=BOUNDARY_BIN_TRIALS,
    recovery_window=RECOVERY_WINDOW_TRIALS,
):
    """Return binned traces and reset/recovery metrics for valid boundaries."""

    aligned_rows = []
    recovery_rows = []
    recovered_column = f"RecoveredWithin{post_trials}"

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

        aligned = pd.DataFrame(
            {
                "AlignedTrial": np.concatenate(
                    (
                        np.arange(-len(previous_rewarded), 0),
                        np.arange(len(next_rewarded)),
                    )
                ),
                "Rewarded": np.concatenate((previous_rewarded, next_rewarded)),
            }
        )
        aligned["AlignedTrialStart"] = (
            aligned["AlignedTrial"] // bin_trials
        ) * bin_trials

        for bin_start, values in aligned.groupby("AlignedTrialStart", sort=True):
            bin_start = int(bin_start)
            aligned_rows.append(
                metadata
                | {
                    "NAvailablePreTrials": len(previous_rewarded),
                    "NAvailablePostTrials": len(next_rewarded),
                    "Period": "Pre" if bin_start < 0 else "Post",
                    "AlignedTrialStart": bin_start,
                    "AlignedTrialEnd": bin_start + bin_trials - 1,
                    "AlignedTrial": bin_start + (bin_trials - 1) / 2,
                    "NTrials": len(values),
                    "Performance": float(values["Rewarded"].mean()),
                }
            )

        baseline = float(np.mean(previous_rewarded))
        initial_post_performance = float(
            np.mean(next_rewarded[:recovery_window])
        )
        absolute_drop, percent_drop, percent_drop_valid = drop_metrics(
            baseline, initial_post_performance
        )
        trailing_rate = (
            pd.Series(next_rewarded)
            .rolling(recovery_window, min_periods=recovery_window)
            .mean()
            .to_numpy()
        )
        recovered_indices = np.flatnonzero(trailing_rate >= baseline)
        recovery_trial = (
            int(recovered_indices[0]) if len(recovered_indices) else np.nan
        )

        recovery_rows.append(
            metadata
            | {
                "NPreBaselineTrials": len(previous_rewarded),
                "NPostTrials": len(next_rewarded),
                "PreBoundaryBaseline": baseline,
                "InitialPostPerformance": initial_post_performance,
                "DropMagnitude": absolute_drop,
                "AbsoluteDrop": absolute_drop,
                "PercentDrop": percent_drop,
                "PercentDropValid": percent_drop_valid,
                "RecoveryTrial": recovery_trial,
                recovered_column: bool(len(recovered_indices)),
            }
        )

    return pd.DataFrame(aligned_rows), recovery_rows


def identity_columns():
    return [
        "Animal",
        "Genotype",
        "Gender",
        "Protocol",
        "BoundaryNumber",
        "PreviousSession",
        "NextSession",
        "PreviousSessionIndex",
        "NextSessionIndex",
        "PreviousDate",
        "NextDate",
        "PreviousProtocolDay",
        "NextProtocolDay",
    ]


def analyze_window(model, pre_trials, post_trials):
    aligned_rows = []
    recovery_rows = []
    animal_info = model.data_index.drop_duplicates("Animal").set_index("Animal")

    for animal in model.data_index["Animal"].unique():
        genotype = animal_info.loc[animal, "Genotype"]
        gender = animal_info.loc[animal, "Gender"]
        for protocol in ("AB", "CD"):
            session_frames = load_protocol_sessions(model, animal, protocol)
            aligned, recovery = boundary_aligned_analysis(
                session_frames,
                pre_trials=pre_trials,
                post_trials=post_trials,
            )
            identity = {
                "Animal": animal,
                "Genotype": genotype,
                "Gender": gender,
                "Protocol": protocol,
            }
            for row in aligned.to_dict("records"):
                aligned_rows.append(identity | row)
            for row in recovery:
                recovery_rows.append(identity | row)

    aligned_columns = identity_columns() + [
        "NAvailablePreTrials",
        "NAvailablePostTrials",
        "Period",
        "AlignedTrialStart",
        "AlignedTrialEnd",
        "AlignedTrial",
        "NTrials",
        "Performance",
    ]
    recovery_columns = identity_columns() + [
        "NPreBaselineTrials",
        "NPostTrials",
        "PreBoundaryBaseline",
        "InitialPostPerformance",
        "DropMagnitude",
        "AbsoluteDrop",
        "PercentDrop",
        "PercentDropValid",
        "RecoveryTrial",
        f"RecoveredWithin{post_trials}",
    ]
    return (
        pd.DataFrame(aligned_rows, columns=aligned_columns),
        pd.DataFrame(recovery_rows, columns=recovery_columns),
    )


def plot_boundary_aligned_recovery(
    aligned_summary, protocol, output_dir, pre_trials, post_trials
):
    protocol_df = aligned_summary[aligned_summary["Protocol"] == protocol].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"WT": "tab:blue", "HET": "tab:orange"}
    plotted = False

    if not protocol_df.empty:
        protocol_df["_plot_genotype"] = (
            protocol_df["Genotype"].astype(str).str.upper()
        )
        for genotype in ("WT", "HET"):
            genotype_df = protocol_df[
                protocol_df["_plot_genotype"] == genotype
            ]
            if genotype_df.empty:
                continue
            curve = (
                genotype_df.groupby("AlignedTrial", as_index=False)
                .agg(
                    MeanPerformance=("Performance", "mean"),
                    SEM=("Performance", "sem"),
                    NBoundaries=("Performance", "size"),
                )
                .sort_values("AlignedTrial")
            )
            x = curve["AlignedTrial"].to_numpy(dtype=float)
            mean = curve["MeanPerformance"].to_numpy(dtype=float)
            sem = curve["SEM"].to_numpy(dtype=float)
            ax.plot(x, mean, color=colors[genotype], linewidth=2, label=genotype)
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=colors[genotype],
                alpha=0.2,
                linewidth=0,
            )
            plotted = True

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlim(-pre_trials, post_trials)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial relative to session boundary")
    ax.set_ylabel("Mean P(correct)")
    ax.set_title(
        f"{protocol}: boundary-aligned recovery "
        f"({window_label(pre_trials, post_trials)}, mean +/- SEM)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    if plotted:
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid WT/HET session boundaries",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{protocol}_boundary_aligned_recovery.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def normalize_metrics(metrics, pre_trials, post_trials):
    normalized = metrics.copy()
    recovered_column = f"RecoveredWithin{post_trials}"
    normalized["WindowConfig"] = window_label(pre_trials, post_trials)
    normalized["Recovered"] = normalized[recovered_column].map(
        lambda value: str(value).strip().lower() == "true"
        if not isinstance(value, (bool, np.bool_))
        else bool(value)
    )
    baseline = pd.to_numeric(normalized["PreBoundaryBaseline"], errors="coerce")
    initial_post = pd.to_numeric(
        normalized["InitialPostPerformance"], errors="coerce"
    )
    normalized["AbsoluteDrop"] = pd.to_numeric(
        normalized["DropMagnitude"], errors="coerce"
    )
    normalized["PercentDropValid"] = (
        baseline.notna()
        & initial_post.notna()
        & ~np.isclose(
            baseline.to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=PERCENT_BASELINE_ATOL,
        )
    )
    normalized["PercentDrop"] = np.where(
        normalized["PercentDropValid"],
        100.0 * (baseline - initial_post) / baseline,
        np.nan,
    )
    return normalized


def summarize_metrics(all_metrics):
    def summarize_group(group):
        drops = pd.to_numeric(group["DropMagnitude"], errors="coerce")
        percent_drops = pd.to_numeric(group["PercentDrop"], errors="coerce")
        percent_valid = group["PercentDropValid"].astype(bool)
        valid_percent_drops = percent_drops[percent_valid]
        baseline = pd.to_numeric(
            group["PreBoundaryBaseline"], errors="coerce"
        )
        initial_post = pd.to_numeric(
            group["InitialPostPerformance"], errors="coerce"
        )
        recovery = pd.to_numeric(group["RecoveryTrial"], errors="coerce")
        recovered = group["Recovered"].astype(bool)
        return pd.Series(
            {
                "NBoundaries": int(len(group)),
                "MeanDrop": drops.mean(),
                "MedianDrop": drops.median(),
                "SDDrop": drops.std(),
                "FractionRecovered": recovered.mean(),
                "MedianRecoveryTrial": recovery.median(),
                "MeanRecoveryTrial": recovery.mean(),
                "FractionPositiveDrop": (drops > 0).mean(),
                "NPercentDropValid": int(percent_valid.sum()),
                "MeanPercentDrop": valid_percent_drops.mean(),
                "MedianPercentDrop": valid_percent_drops.median(),
                "SDPercentDrop": valid_percent_drops.std(),
                "Q1PercentDrop": valid_percent_drops.quantile(0.25),
                "Q3PercentDrop": valid_percent_drops.quantile(0.75),
                "MeanPreBoundaryBaseline": baseline.mean(),
                "MedianPreBoundaryBaseline": baseline.median(),
                "MeanInitialPostPerformance": initial_post.mean(),
                "MedianInitialPostPerformance": initial_post.median(),
            }
        )

    genotype = (
        all_metrics.groupby(
            ["WindowConfig", "Protocol", "Genotype"], dropna=False, sort=False
        )
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    genotype.insert(1, "SummaryLevel", "ProtocolGenotype")
    overall = (
        all_metrics.groupby(["WindowConfig", "Protocol"], sort=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    overall.insert(1, "SummaryLevel", "ProtocolOverall")
    overall.insert(3, "Genotype", "Overall")
    return pd.concat((genotype, overall), ignore_index=True)


def build_difference_table(summary):
    group_summary = summary[summary["SummaryLevel"] == "ProtocolGenotype"]
    pairs = (
        ("pre200/post200", "pre300/post300"),
        ("pre300/post300", "pre300/post500"),
        ("pre200/post200", "pre300/post500"),
    )
    rows = []
    for from_window, to_window in pairs:
        left = group_summary[group_summary["WindowConfig"] == from_window]
        right = group_summary[group_summary["WindowConfig"] == to_window]
        merged = left.merge(
            right,
            on=["Protocol", "Genotype"],
            suffixes=("_From", "_To"),
        )
        for row in merged.itertuples(index=False):
            for statistic in SUMMARY_STATISTICS:
                from_value = getattr(row, f"{statistic}_From")
                to_value = getattr(row, f"{statistic}_To")
                rows.append(
                    {
                        "Comparison": f"{from_window} vs {to_window}",
                        "FromWindow": from_window,
                        "ToWindow": to_window,
                        "Protocol": row.Protocol,
                        "Genotype": row.Genotype,
                        "Statistic": statistic,
                        "FromValue": from_value,
                        "ToValue": to_value,
                        "Difference": to_value - from_value,
                    }
                )
    return pd.DataFrame(rows)


def plot_window_comparison(summary, output_path):
    group_summary = summary[summary["SummaryLevel"] == "ProtocolGenotype"].copy()
    window_order = [
        label
        for label in ("pre200/post200", "pre300/post300", "pre300/post500")
        if label in set(group_summary["WindowConfig"])
    ]
    metrics = (
        ("MeanDrop", "Mean DropMagnitude"),
        ("FractionRecovered", "Fraction recovered"),
        ("MedianRecoveryTrial", "Median RecoveryTrial"),
    )
    colors = {"WT": "tab:blue", "HET": "tab:orange"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    for row_index, protocol in enumerate(("AB", "CD")):
        for column_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column_index]
            protocol_df = group_summary[group_summary["Protocol"] == protocol]
            for genotype in ("WT", "HET"):
                genotype_df = protocol_df[
                    protocol_df["Genotype"].astype(str).str.upper() == genotype
                ].set_index("WindowConfig")
                values = [
                    genotype_df.at[label, metric]
                    if label in genotype_df.index
                    else np.nan
                    for label in window_order
                ]
                ax.plot(
                    np.arange(len(window_order)),
                    values,
                    marker="o",
                    linewidth=1.8,
                    color=colors[genotype],
                    label=genotype,
                )
            if metric == "MeanDrop":
                ax.axhline(0, color="black", linestyle=":", linewidth=1)
            ax.set_xticks(np.arange(len(window_order)), window_order, rotation=20)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{protocol}: {ylabel}")
            ax.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                ax.legend(frameon=False)
    fig.suptitle("Session-boundary window sensitivity", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_vs_percent_boundary_drop(all_metrics, output_path):
    """Compare boundary-level absolute and relative drops for four groups."""

    groups = (("AB", "WT"), ("AB", "HET"), ("CD", "WT"), ("CD", "HET"))
    labels = [f"{protocol} {genotype}" for protocol, genotype in groups]
    windows = [
        label
        for label in ("pre200/post200", "pre300/post300", "pre300/post500")
        if label in set(all_metrics["WindowConfig"])
    ]
    colors = {
        "pre200/post200": "tab:blue",
        "pre300/post300": "tab:orange",
        "pre300/post500": "tab:green",
    }
    offsets = np.linspace(-0.24, 0.24, len(windows))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, ylabel in (
        (axes[0], "AbsoluteDrop", "Absolute drop (proportion / percentage points)"),
        (axes[1], "PercentDrop", "Relative drop from baseline (%)"),
    ):
        for group_index, (protocol, genotype) in enumerate(groups):
            for offset, window in zip(offsets, windows):
                values = pd.to_numeric(
                    all_metrics.loc[
                        (all_metrics["WindowConfig"] == window)
                        & (all_metrics["Protocol"] == protocol)
                        & (
                            all_metrics["Genotype"].astype(str).str.upper()
                            == genotype
                        ),
                        metric,
                    ],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                center = group_index + offset
                jitter = rng.uniform(-0.045, 0.045, size=len(values))
                ax.scatter(
                    center + jitter,
                    values,
                    s=16,
                    alpha=0.25,
                    color=colors[window],
                    linewidths=0,
                )
                ax.scatter(
                    center,
                    values.mean(),
                    s=75,
                    marker="D",
                    color=colors[window],
                    edgecolor="black",
                    linewidth=0.7,
                    label=window if group_index == 0 else None,
                    zorder=3,
                )
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(np.arange(len(groups)), labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("Absolute DropMagnitude", fontsize=14)
    axes[1].set_title("Relative PercentDrop", fontsize=14)
    axes[0].legend(title="Window", frameon=False, fontsize=9, title_fontsize=10)
    fig.suptitle(
        "Absolute and relative session-boundary drops", fontsize=16, y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_percent_drop_window_sensitivity(summary, output_path):
    """Plot mean and median group-level PercentDrop across windows."""

    group_summary = summary[summary["SummaryLevel"] == "ProtocolGenotype"].copy()
    windows = [
        label
        for label in ("pre200/post200", "pre300/post300", "pre300/post500")
        if label in set(group_summary["WindowConfig"])
    ]
    groups = (
        ("AB", "WT", "tab:blue"),
        ("AB", "HET", "tab:cyan"),
        ("CD", "WT", "tab:orange"),
        ("CD", "HET", "tab:red"),
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(windows))

    for protocol, genotype, color in groups:
        selected = group_summary[
            (group_summary["Protocol"] == protocol)
            & (group_summary["Genotype"].astype(str).str.upper() == genotype)
        ].set_index("WindowConfig")
        means = [
            selected.at[window, "MeanPercentDrop"]
            if window in selected.index
            else np.nan
            for window in windows
        ]
        medians = [
            selected.at[window, "MedianPercentDrop"]
            if window in selected.index
            else np.nan
            for window in windows
        ]
        ax.plot(
            x,
            means,
            color=color,
            marker="o",
            linewidth=2,
            label=f"{protocol} {genotype} mean",
        )
        ax.plot(
            x,
            medians,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"{protocol} {genotype} median",
        )

    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, windows, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Relative drop from pre-boundary baseline (%)", fontsize=11)
    ax.set_title(
        "PercentDrop window sensitivity: group mean and median", fontsize=14
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def read_reference_metrics(reference_output):
    path = reference_output / "boundary_recovery_metrics.csv"
    if not path.is_file():
        return None
    metrics = pd.read_csv(path)
    required = {
        "Protocol",
        "Genotype",
        "DropMagnitude",
        "RecoveryTrial",
        "RecoveredWithin500",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")
    return normalize_metrics(metrics, *REFERENCE_WINDOW)


def expected_output_paths(output_dir):
    paths = [
        output_dir / "boundary_window_sensitivity_summary.csv",
        output_dir / "boundary_window_sensitivity_difference.csv",
        output_dir / "boundary_window_sensitivity_comparison.png",
        output_dir / "absolute_vs_percent_boundary_drop.png",
        output_dir / "percent_drop_window_sensitivity.png",
        output_dir / "README_boundary_window_sensitivity.txt",
    ]
    for pre_trials, post_trials in WINDOW_CONFIGS:
        config_dir = output_dir / f"pre{pre_trials}_post{post_trials}"
        paths.extend(
            [
                config_dir / "boundary_aligned_summary.csv",
                config_dir / "boundary_recovery_metrics.csv",
                config_dir / "AB_boundary_aligned_recovery.png",
                config_dir / "CD_boundary_aligned_recovery.png",
            ]
        )
    return paths


def validate_output_separation(output_dir, reference_output):
    if output_dir == reference_output or reference_output in output_dir.parents:
        raise ValueError(
            "Sensitivity output must be separate from the read-only reference output"
        )


def write_readme(output_dir, reference_available):
    reference_note = (
        "The existing pre=300/post=500 boundary_recovery_metrics.csv was read "
        "as reference data for the cross-window comparison. It was not recomputed "
        "or overwritten."
        if reference_available
        else "No existing pre=300/post=500 boundary_recovery_metrics.csv was found, "
        "so that window is omitted from the cross-window comparison."
    )
    text = f"""Boundary-window sensitivity analysis

This is a sensitivity analysis of analysis-window choice for the session-boundary
reset/recovery result. New analyses use pre=200/post=200 and pre=300/post=300.
Each boundary is aligned to trial 0, the first trial of the next session. Negative
trials come from the end of the previous session. Aligned plots and CSVs use
non-overlapping 20-trial bins, WT and HET are shown separately, and no additional
smoothing or normalization is applied.

The immediate-drop definition is unchanged: PreBoundaryBaseline is the mean raw
binary reward over all available pre-boundary trials within the configured window;
InitialPostPerformance is the mean over the first 50 available post-boundary trials;
DropMagnitude = PreBoundaryBaseline - InitialPostPerformance.
AbsoluteDrop is an explicit copy of this same absolute proportion difference.
A value such as 0.18 is an 18-percentage-point absolute drop; DropMagnitude and
AbsoluteDrop are not percent changes. PercentDrop is calculated separately for
each boundary as 100 * AbsoluteDrop / PreBoundaryBaseline before any group
aggregation. Positive PercentDrop means performance decreased, while negative
PercentDrop means it increased. PercentDropValid is false and PercentDrop is NaN
when the baseline is non-finite or has absolute value <= {PERCENT_BASELINE_ATOL:g}.

The recovery definition is unchanged: RecoveryTrial is the earliest zero-based
post-boundary trial where the trailing 50-trial raw reward rate reaches or exceeds
PreBoundaryBaseline. A full trailing-50 window is required, so the earliest possible
RecoveryTrial is 49. RecoveredWithin200 and RecoveredWithin300 indicate whether this
occurs within their respective post windows.

Short sessions contribute every available trial up to the configured limit. Fewer
than 50 post-boundary trials are still used for InitialPostPerformance, but recovery
remains undefined because a full trailing-50 window is unavailable.

{reference_note}

Agreement across windows supports robustness of the session-boundary reset/recovery
conclusion. Source behavioral data and original analysis outputs are read only.
"""
    (output_dir / "README_boundary_window_sensitivity.txt").write_text(
        text, encoding="utf-8"
    )


def selected_summary_lines(summary):
    available = [
        label
        for label in ("pre200/post200", "pre300/post300", "pre300/post500")
        if label in set(summary["WindowConfig"])
    ]
    selections = (("AB", "WT"), ("AB", "HET"), ("CD", "WT"), ("CD", "HET"))
    lines = [
        "Group | WindowConfig | N | MeanPreBoundaryBaseline | "
        "MeanAbsoluteDrop | MeanPercentDrop | MedianPercentDrop | "
        "FractionRecovered | MedianRecoveryTrial"
    ]
    for protocol, genotype in selections:
        selected = summary[
            (summary["Protocol"] == protocol)
            & (summary["Genotype"].astype(str).str.upper() == genotype.upper())
            & (summary["SummaryLevel"] == "ProtocolGenotype")
        ].set_index("WindowConfig")
        for label in available:
            if label not in selected.index:
                continue
            row = selected.loc[label]
            lines.append(
                f"{protocol} {genotype} | {label} | {int(row['NBoundaries'])} | "
                f"{row['MeanPreBoundaryBaseline']:.4f} | {row['MeanDrop']:.4f} | "
                f"{row['MeanPercentDrop']:.2f} | {row['MedianPercentDrop']:.2f} | "
                f"{row['FractionRecovered']:.3f} | "
                f"{row['MedianRecoveryTrial']:.1f}"
            )
    return lines


def qualitative_stability(summary):
    group = summary[summary["SummaryLevel"] == "ProtocolGenotype"]
    overall = summary[summary["SummaryLevel"] == "ProtocolOverall"]
    windows = set(summary["WindowConfig"])
    stable_relative = True
    cd_exceeds_ab = True
    het_cd_strongest = True
    for window in windows:
        window_groups = group[group["WindowConfig"] == window]
        stable_relative &= bool((window_groups["MeanPercentDrop"] > 0).all())
        window_overall = overall[overall["WindowConfig"] == window].set_index(
            "Protocol"
        )
        cd_exceeds_ab &= bool(
            window_overall.at["CD", "MeanPercentDrop"]
            > window_overall.at["AB", "MeanPercentDrop"]
        )
        cd_het = window_groups[
            (window_groups["Protocol"] == "CD")
            & (window_groups["Genotype"].astype(str).str.upper() == "HET")
        ]["MeanPercentDrop"].iloc[0]
        het_cd_strongest &= bool(cd_het == window_groups["MeanPercentDrop"].max())
    return (
        f"relative-drop direction stable={stable_relative}; "
        f"CD overall > AB overall={cd_exceeds_ab}; "
        f"CD HET has largest group MeanPercentDrop={het_cd_strongest}. "
        "Descriptive comparisons only; no genotype-effect inference is made."
    )


def run_synthetic_checks():
    def session(rewards, number, date):
        return pd.DataFrame(
            {
                "reward": rewards,
                "_session_number": number,
                "_session_index": number + 10,
                "_date": date,
                "_protocol_day": number,
                "_trial_in_session": np.arange(1, len(rewards) + 1),
            }
        )

    previous = session([1, 1, 1, 1, 0] * 60, 1, "20230101")
    following = session([1] * 30 + [0] * 20 + [1] * 250, 2, "20230102")
    for pre_trials, post_trials in WINDOW_CONFIGS:
        _, rows = boundary_aligned_analysis(
            [previous, following], pre_trials, post_trials
        )
        metric = rows[0]
        assert np.isclose(metric["PreBoundaryBaseline"], 0.8)
        assert np.isclose(metric["InitialPostPerformance"], 0.6)
        assert np.isclose(metric["DropMagnitude"], 0.2)
        assert metric["AbsoluteDrop"] == metric["DropMagnitude"]
        assert np.isclose(metric["PercentDrop"], 25.0)
        assert metric["PercentDropValid"]
        assert metric["RecoveryTrial"] == 89
        assert metric[f"RecoveredWithin{post_trials}"]
    print(
        "Synthetic checks passed in both windows: baseline=0.80, initial=0.60, "
        "AbsoluteDrop=0.20, PercentDrop=25.0%, RecoveryTrial=89"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--reference-output")
    parser.add_argument("--strain")
    parser.add_argument("--synthetic-test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.synthetic_test_only:
        run_synthetic_checks()
        return
    if not args.root_dir or not args.output_dir or not args.reference_output:
        raise SystemExit(
            "--root-dir, --output-dir, and --reference-output are required"
        )

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_output = Path(args.reference_output).expanduser().resolve()
    validate_output_separation(output_dir, reference_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    strain = args.strain or root_dir.name
    model = ReadOnlyBehDataOdor(str(root_dir), strain)

    normalized_metrics = []
    for pre_trials, post_trials in WINDOW_CONFIGS:
        config_dir = output_dir / f"pre{pre_trials}_post{post_trials}"
        config_dir.mkdir(parents=True, exist_ok=True)
        aligned, recovery = analyze_window(model, pre_trials, post_trials)
        aligned.to_csv(config_dir / "boundary_aligned_summary.csv", index=False)
        recovery.to_csv(config_dir / "boundary_recovery_metrics.csv", index=False)
        for protocol in ("AB", "CD"):
            plot_boundary_aligned_recovery(
                aligned, protocol, config_dir, pre_trials, post_trials
            )
        normalized_metrics.append(
            normalize_metrics(recovery, pre_trials, post_trials)
        )

    reference = read_reference_metrics(reference_output)
    if reference is not None:
        normalized_metrics.append(reference)
    all_metrics = pd.concat(normalized_metrics, ignore_index=True, sort=False)
    summary = summarize_metrics(all_metrics)
    difference = build_difference_table(summary)
    summary.to_csv(
        output_dir / "boundary_window_sensitivity_summary.csv", index=False
    )
    difference.to_csv(
        output_dir / "boundary_window_sensitivity_difference.csv", index=False
    )
    plot_window_comparison(
        summary, output_dir / "boundary_window_sensitivity_comparison.png"
    )
    plot_absolute_vs_percent_boundary_drop(
        all_metrics, output_dir / "absolute_vs_percent_boundary_drop.png"
    )
    plot_percent_drop_window_sensitivity(
        summary, output_dir / "percent_drop_window_sensitivity.png"
    )
    write_readme(output_dir, reference is not None)

    print(f"Boundary-window sensitivity analysis complete: {output_dir}")
    print("\n".join(selected_summary_lines(summary)))
    print("Descriptive conclusions: " + qualitative_stability(summary))


if __name__ == "__main__":
    main()
