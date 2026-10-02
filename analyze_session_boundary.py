"""Diagnose discontinuities at behavioral-session boundaries.

This script is an external, read-only analysis wrapper. It does not modify
BehDataOdor.find_eureka or any source behavior CSVs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

# This diagnostic script is headless. Some mouscope modules explicitly
# request QtAgg, which is unavailable in the macOS analysis environment.
_original_matplotlib_use = matplotlib.use


def _headless_matplotlib_use(backend, *args, **kwargs):
    if str(backend).lower() in {"qtagg", "qt5agg", "qt6agg"}:
        backend = "Agg"
    return _original_matplotlib_use(backend, *args, **kwargs)


matplotlib.use = _headless_matplotlib_use
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavioral_pipeline import BehDataOdor


BOUNDARY_PRE_TRIALS = 300
BOUNDARY_POST_TRIALS = 500
BOUNDARY_BIN_TRIALS = 20
RECOVERY_WINDOW_TRIALS = 50


class ReadOnlyBehDataOdor(BehDataOdor):
    """Use existing processed behavior CSVs without creating missing files."""

    def load_data(self):
        for idx, row in self.data_index.iterrows():
            csv_path = os.path.join(
                row["AnalysisPath"],
                f'{row["Date"]}_{row["Protocol"]}{row["ProtocolDay"]}.csv',
            )
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    "Processed behavior CSV is missing; read-only analysis will not "
                    f"create it: {csv_path}"
                )
            self.data_index.at[idx, "BehCSV"] = csv_path


def load_protocol_sessions(model, animal, protocol):
    sessions = model.data_index[model.data_index["Animal"] == animal].copy()

    transition = sessions[sessions["Protocol"].str.contains("AB-CD")]
    first_transition_idx = transition.index[0] if not transition.empty else None

    if protocol == "AB":
        selected = (
            sessions.loc[: first_transition_idx - 1]
            if first_transition_idx is not None
            else sessions
        )
    elif protocol == "CD":
        selected = sessions[
            (sessions["Protocol"] == "AB-CD")
            & (sessions["ProtocolDay"] <= 3)
        ]
    else:
        raise ValueError(protocol)

    frames = []

    for session_number, (session_idx, row) in enumerate(
        selected.iterrows(), start=1
    ):
        frame = pd.read_csv(row["BehCSV"])

        frame = frame[~np.isnan(frame["actions"])].copy()

        if protocol == "CD":
            frame = frame[frame["schedule"] > 2].copy()

        if frame.empty:
            continue

        frame["_session_number"] = session_number
        frame["_session_index"] = int(session_idx)
        frame["_date"] = str(row["Date"])
        frame["_protocol_day"] = row["ProtocolDay"]
        frame["_trial_in_session"] = np.arange(1, len(frame) + 1)

        frames.append(frame.reset_index(drop=True))

    return frames


def rewarded_vector(frame):
    reward = frame["reward"].fillna(0).replace([2, 3], 1)
    return (reward > 0).astype(float).to_numpy()


def boundary_aligned_analysis(
    session_frames,
    pre_trials=BOUNDARY_PRE_TRIALS,
    post_trials=BOUNDARY_POST_TRIALS,
    bin_trials=BOUNDARY_BIN_TRIALS,
    recovery_window=RECOVERY_WINDOW_TRIALS,
):
    """Return binned boundary traces and recovery metrics for valid pairs."""

    aligned_rows = []
    recovery_rows = []

    for boundary_number, (previous, next_session) in enumerate(
        zip(session_frames[:-1], session_frames[1:]),
        start=1,
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

        previous_session = int(previous["_session_number"].iloc[-1])
        next_session_number = int(next_session["_session_number"].iloc[0])
        previous_date = previous["_date"].iloc[-1]
        next_date = next_session["_date"].iloc[0]
        previous_protocol_day = previous["_protocol_day"].iloc[-1]
        next_protocol_day = next_session["_protocol_day"].iloc[0]

        aligned_trials = np.concatenate(
            [
                np.arange(-len(previous_rewarded), 0),
                np.arange(len(next_rewarded)),
            ]
        )
        aligned_rewarded = np.concatenate(
            [previous_rewarded, next_rewarded]
        )
        aligned = pd.DataFrame(
            {
                "AlignedTrial": aligned_trials,
                "Rewarded": aligned_rewarded,
            }
        )
        aligned["AlignedTrialStart"] = (
            aligned["AlignedTrial"] // bin_trials
        ) * bin_trials

        for bin_start, values in aligned.groupby(
            "AlignedTrialStart",
            sort=True,
        ):
            bin_start = int(bin_start)
            aligned_rows.append(
                {
                    "BoundaryNumber": boundary_number,
                    "PreviousSession": previous_session,
                    "NextSession": next_session_number,
                    "PreviousDate": previous_date,
                    "NextDate": next_date,
                    "PreviousProtocolDay": previous_protocol_day,
                    "NextProtocolDay": next_protocol_day,
                    "Period": "Pre" if bin_start < 0 else "Post",
                    "AlignedTrialStart": bin_start,
                    "AlignedTrialEnd": bin_start + bin_trials - 1,
                    "AlignedTrial": bin_start + (bin_trials - 1) / 2,
                    "NTrials": len(values),
                    "Performance": float(values["Rewarded"].mean()),
                }
            )

        baseline = float(np.mean(previous_rewarded))
        initial_post = next_rewarded[:recovery_window]
        initial_post_performance = float(np.mean(initial_post))
        trailing_rate = (
            pd.Series(next_rewarded)
            .rolling(
                recovery_window,
                min_periods=recovery_window,
            )
            .mean()
            .to_numpy()
        )
        recovered = np.flatnonzero(trailing_rate >= baseline)
        recovery_trial = int(recovered[0]) if len(recovered) else np.nan

        recovery_rows.append(
            {
                "BoundaryNumber": boundary_number,
                "PreviousSession": previous_session,
                "NextSession": next_session_number,
                "PreviousDate": previous_date,
                "NextDate": next_date,
                "PreviousProtocolDay": previous_protocol_day,
                "NextProtocolDay": next_protocol_day,
                "NPreBaselineTrials": len(previous_rewarded),
                "NPostTrials": len(next_rewarded),
                "PreBoundaryBaseline": baseline,
                "InitialPostPerformance": initial_post_performance,
                "DropMagnitude": baseline - initial_post_performance,
                "RecoveryTrial": recovery_trial,
                "RecoveredWithin500": bool(len(recovered)),
            }
        )

    return pd.DataFrame(aligned_rows), recovery_rows


def smooth_learning_curve(
    rewarded,
    running_window=60,
    smooth_window=500,
):
    """Replicate the smoothing used in find_eureka()."""

    n_trials = len(rewarded)

    running = np.full(n_trials, np.nan, dtype=float)

    if n_trials >= running_window:
        csum = np.empty(n_trials + 1, dtype=float)
        csum[0] = 0

        np.cumsum(rewarded, out=csum[1:])

        running[: n_trials - running_window + 1] = (
            csum[running_window:]
            - csum[:-running_window]
        ) / running_window

    return (
        pd.Series(running)
        .rolling(
            smooth_window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


def concatenate_with_boundaries(session_frames):
    if not session_frames:
        return pd.DataFrame(), []

    boundaries = []
    cumulative = 0

    for frame in session_frames[:-1]:
        cumulative += len(frame)
        boundaries.append(cumulative)

    concatenated = pd.concat(
        session_frames,
        ignore_index=True,
    )

    concatenated["_global_trial"] = np.arange(
        1,
        len(concatenated) + 1,
    )

    return concatenated, boundaries


def boundary_metrics(
    concatenated,
    boundaries,
    window_trials,
):
    if concatenated.empty or "reward" not in concatenated.columns:
        return []

    rewarded = rewarded_vector(concatenated)

    rows = []

    for boundary_number, boundary in enumerate(
        boundaries,
        start=1,
    ):
        before_start = max(
            0,
            boundary - window_trials,
        )

        after_end = min(
            len(rewarded),
            boundary + window_trials,
        )

        before = rewarded[
            before_start:boundary
        ]

        after = rewarded[
            boundary:after_end
        ]

        prev_row = concatenated.iloc[
            boundary - 1
        ]

        next_row = concatenated.iloc[
            boundary
        ]

        before_perf = (
            float(np.mean(before))
            if len(before)
            else np.nan
        )

        after_perf = (
            float(np.mean(after))
            if len(after)
            else np.nan
        )

        rows.append(
            {
                "BoundaryNumber": boundary_number,
                "BoundaryGlobalTrial": boundary,

                "PreviousSession":
                    int(prev_row["_session_number"]),

                "NextSession":
                    int(next_row["_session_number"]),

                "PreviousDate":
                    prev_row["_date"],

                "NextDate":
                    next_row["_date"],

                "PreviousProtocolDay":
                    prev_row["_protocol_day"],

                "NextProtocolDay":
                    next_row["_protocol_day"],

                "NBefore": len(before),
                "NAfter": len(after),

                "BeforePerformance":
                    before_perf,

                "AfterPerformance":
                    after_perf,

                "PerformanceChange":
                    after_perf - before_perf,

                "PerformanceDrop":
                    before_perf - after_perf,
            }
        )

    return rows


def plot_animal_protocol(
    animal,
    genotype,
    protocol,
    concatenated,
    boundaries,
    output_dir,
    running_window,
    smooth_window,
):
    if concatenated.empty:
        return

    rewarded = rewarded_vector(
        concatenated
    )

    smoothed = smooth_learning_curve(
        rewarded,
        running_window,
        smooth_window,
    )

    x = np.arange(
        1,
        len(smoothed) + 1,
    )

    valid = np.isfinite(smoothed)

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        x[valid],
        smoothed[valid],
        color="black",
        linewidth=2,
        label="Concatenated curve",
    )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    for i, boundary in enumerate(
        boundaries,
        start=1,
    ):
        ax.axvline(
            boundary + 0.5,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=(
                "Session boundary"
                if i == 1
                else None
            ),
        )

    ax.set_xlabel(
        "Concatenated trial"
    )

    ax.set_ylabel(
        "P(correct)"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        f"{animal} ({genotype}) {protocol}: "
        "learning curve with session boundaries"
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    animal_dir = (
        output_dir / str(animal)
    )

    animal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        animal_dir
        / f"{animal}_{protocol}_session_boundaries.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_boundary_change_distribution(
    summary,
    output_dir,
):
    if summary.empty:
        return

    for protocol in summary[
        "Protocol"
    ].dropna().unique():

        protocol_df = summary[
            summary["Protocol"] == protocol
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        genotypes = list(
            protocol_df[
                "Genotype"
            ]
            .dropna()
            .unique()
        )

        positions = np.arange(
            len(genotypes)
        )

        for pos, genotype in zip(
            positions,
            genotypes,
        ):
            values = (
                protocol_df.loc[
                    protocol_df[
                        "Genotype"
                    ] == genotype,
                    "PerformanceChange",
                ]
                .dropna()
                .to_numpy()
            )

            if len(values):
                jitter = (
                    np.linspace(
                        -0.08,
                        0.08,
                        len(values),
                    )
                    if len(values) > 1
                    else np.array([0.0])
                )

                ax.scatter(
                    np.full(
                        len(values),
                        pos,
                    )
                    + jitter,
                    values,
                    alpha=0.75,
                )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        ax.set_xticks(
            positions,
            genotypes,
        )

        ax.set_ylabel(
            "After - before performance"
        )

        ax.set_title(
            f"{protocol}: session-boundary performance change"
        )

        ax.spines[
            ["top", "right"]
        ].set_visible(False)

        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"{protocol}_boundary_performance_change.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_boundary_aligned_recovery(
    aligned_summary,
    protocol,
    output_dir,
):
    protocol_df = aligned_summary[
        aligned_summary["Protocol"] == protocol
    ].copy()

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

            ax.plot(
                x,
                mean,
                color=colors[genotype],
                linewidth=2,
                label=genotype,
            )
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=colors[genotype],
                alpha=0.2,
                linewidth=0,
            )
            plotted = True

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Session boundary",
    )
    ax.axhline(
        0.5,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )
    ax.set_xlim(-BOUNDARY_PRE_TRIALS, BOUNDARY_POST_TRIALS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial relative to session boundary")
    ax.set_ylabel("Mean P(correct)")
    ax.set_title(
        f"{protocol}: boundary-aligned recovery (mean +/- SEM)"
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--strain",
    )

    parser.add_argument(
        "--boundary-window",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--running-window",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=500,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = (
        Path(args.root_dir)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    strain = (
        args.strain
        or root_dir.name
    )

    model = ReadOnlyBehDataOdor(
        str(root_dir),
        strain,
    )

    animal_info = (
        model.data_index
        .drop_duplicates("Animal")
        .set_index("Animal")
    )

    rows = []
    session_rows = []
    aligned_rows = []
    recovery_rows = []

    for animal in model.data_index[
        "Animal"
    ].unique():

        genotype = animal_info.loc[
            animal,
            "Genotype",
        ]

        gender = animal_info.loc[
            animal,
            "Gender",
        ]

        for protocol in (
            "AB",
            "CD",
        ):

            session_frames = (
                load_protocol_sessions(
                    model,
                    animal,
                    protocol,
                )
            )

            (
                concatenated,
                boundaries,
            ) = concatenate_with_boundaries(
                session_frames
            )

            for frame in session_frames:

                rewarded = (
                    rewarded_vector(
                        frame
                    )
                )

                session_rows.append(
                    {
                        "Animal":
                            animal,

                        "Genotype":
                            genotype,

                        "Gender":
                            gender,

                        "Protocol":
                            protocol,

                        "SessionNumber":
                            int(
                                frame[
                                    "_session_number"
                                ].iloc[0]
                            ),

                        "Date":
                            frame[
                                "_date"
                            ].iloc[0],

                        "ProtocolDay":
                            frame[
                                "_protocol_day"
                            ].iloc[0],

                        "NTrials":
                            len(frame),

                        "SessionPerformance":
                            float(
                                np.mean(
                                    rewarded
                                )
                            )
                            if len(rewarded)
                            else np.nan,
                    }
                )

            metrics = boundary_metrics(
                concatenated,
                boundaries,
                args.boundary_window,
            )

            (
                boundary_aligned,
                boundary_recovery,
            ) = boundary_aligned_analysis(session_frames)

            identity = {
                "Animal": animal,
                "Genotype": genotype,
                "Gender": gender,
                "Protocol": protocol,
            }

            for aligned_row in boundary_aligned.to_dict("records"):
                aligned_row.update(identity)
                aligned_rows.append(aligned_row)

            for recovery_row in boundary_recovery:
                recovery_row.update(identity)
                recovery_rows.append(recovery_row)

            for metric in metrics:

                metric.update(
                    {
                        "Animal":
                            animal,

                        "Genotype":
                            genotype,

                        "Gender":
                            gender,

                        "Protocol":
                            protocol,
                    }
                )

                rows.append(
                    metric
                )

            plot_animal_protocol(
                animal,
                genotype,
                protocol,
                concatenated,
                boundaries,
                output_dir,
                args.running_window,
                args.smooth_window,
            )

    boundary_summary = pd.DataFrame(
        rows
    )

    session_summary = pd.DataFrame(
        session_rows
    )

    identity_columns = [
        "Animal",
        "Genotype",
        "Gender",
        "Protocol",
        "BoundaryNumber",
        "PreviousSession",
        "NextSession",
        "PreviousDate",
        "NextDate",
        "PreviousProtocolDay",
        "NextProtocolDay",
    ]

    boundary_aligned_summary = pd.DataFrame(
        aligned_rows,
        columns=identity_columns
        + [
            "Period",
            "AlignedTrialStart",
            "AlignedTrialEnd",
            "AlignedTrial",
            "NTrials",
            "Performance",
        ],
    )

    boundary_recovery_metrics = pd.DataFrame(
        recovery_rows,
        columns=identity_columns
        + [
            "NPreBaselineTrials",
            "NPostTrials",
            "PreBoundaryBaseline",
            "InitialPostPerformance",
            "DropMagnitude",
            "RecoveryTrial",
            "RecoveredWithin500",
        ],
    )

    boundary_summary.to_csv(
        output_dir
        / "session_boundary_summary.csv",
        index=False,
    )

    session_summary.to_csv(
        output_dir
        / "session_performance_summary.csv",
        index=False,
    )

    boundary_aligned_summary.to_csv(
        output_dir / "boundary_aligned_summary.csv",
        index=False,
    )

    boundary_recovery_metrics.to_csv(
        output_dir / "boundary_recovery_metrics.csv",
        index=False,
    )

    for protocol in ("AB", "CD"):
        plot_boundary_aligned_recovery(
            boundary_aligned_summary,
            protocol,
            output_dir,
        )

    if not boundary_summary.empty:

        grouped = (
            boundary_summary
            .groupby(
                [
                    "Protocol",
                    "Genotype",
                ],
                dropna=False,
            )
            .agg(
                NBoundaries=(
                    "PerformanceChange",
                    "size",
                ),

                MeanChange=(
                    "PerformanceChange",
                    "mean",
                ),

                MedianChange=(
                    "PerformanceChange",
                    "median",
                ),

                SDChange=(
                    "PerformanceChange",
                    "std",
                ),

                MeanDrop=(
                    "PerformanceDrop",
                    "mean",
                ),

                FractionDrops=(
                    "PerformanceChange",
                    lambda x:
                        float(
                            np.mean(
                                np.asarray(x) < 0
                            )
                        ),
                ),
            )
            .reset_index()
        )

        grouped.to_csv(
            output_dir
            / "session_boundary_group_summary.csv",
            index=False,
        )

        plot_boundary_change_distribution(
            boundary_summary,
            output_dir,
        )

    note = (
        "This analysis preserves behavioral-session identity before concatenation. "
        "PerformanceChange is computed from raw rewarded-trial fractions in the "
        f"last/first {args.boundary_window} trials surrounding each session boundary. "
        "Positive values mean performance increased across the boundary; negative "
        "values mean performance dropped. Diagnostic learning curves reproduce the "
        f"existing {args.running_window}-trial running reward rate followed by a "
        f"{args.smooth_window}-trial centered rolling mean. Source CSVs are read only.\n"
        "\nBoundary-aligned recovery analysis:\n"
        "Each valid boundary uses raw binary rewarded/correct values from rewarded_vector(), "
        "with the last 300 trials of the previous session aligned to -300 through -1 "
        "and the first 500 trials of the next session aligned to 0 through 499. "
        "The aligned CSV reports non-overlapping 20-trial bins; shorter sessions contribute "
        "only their observed trials and are never shifted or normalized. The pre-boundary "
        "baseline is the mean rewarded/correct performance across all available trials "
        "within -300 to -1; if fewer than 300 trials exist, all available trials in that "
        "range are used. Initial post-boundary performance is the mean of the first 50 "
        "available post-boundary trials. DropMagnitude is pre-boundary baseline minus "
        "initial post-boundary performance. RecoveryTrial is the earliest zero-based "
        "post-boundary trial (0 to 499) at which the trailing 50-trial raw reward rate "
        "reaches or exceeds the pre-boundary baseline. At least 50 post-boundary trials "
        "are therefore required to recover. RecoveredWithin500 is true exactly when such "
        "a RecoveryTrial exists. No additional smoothing or session normalization is "
        "applied. Plot lines are genotype means across boundary bins and shaded regions "
        "are SEM across boundaries.\n"
    )

    (
        output_dir
        / "README_session_boundary_analysis.txt"
    ).write_text(
        note,
        encoding="utf-8",
    )

    print(
        f"Session-boundary analysis complete: "
        f"{output_dir}"
    )

    print(
        f"Boundaries analyzed: "
        f"{len(boundary_summary)}"
    )

    print(
        f"Sessions analyzed: "
        f"{len(session_summary)}"
    )


if __name__ == "__main__":
    main()
