"""QC existing continuous and session-aware sigmoid comparison fits.

This external diagnostic reads already-generated comparison and session-level
summary CSVs. It does not preprocess behavior, refit models, or modify source
behavioral data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTINUOUS_FLAGS = [
    "TauContNearLowerBound",
    "TauContNearUpperBound",
    "KContNearLowerBound",
    "KContNearUpperBound",
]

SESSION_AWARE_FLAGS = [
    "TauSessionNearLowerBound",
    "TauSessionNearUpperBound",
    "KSessionNearLowerBound",
    "KSessionNearUpperBound",
    "ResetANearLowerBound",
    "ResetANearUpperBound",
    "RecoveryLambdaNearLowerBound",
    "RecoveryLambdaNearUpperBound",
]

ALL_BOUNDARY_FLAGS = CONTINUOUS_FLAGS + SESSION_AWARE_FLAGS

COMPARISON_REQUIRED_COLUMNS = {
    "Animal",
    "Genotype",
    "Protocol",
    "NTrials",
    "NSessions",
    "success_cont",
    "success_session",
    "tau_cont",
    "k_cont",
    "tau_session",
    "k_session",
    "reset_A",
    "recovery_lambda",
    "DeltaTau",
    "DeltaK",
    "DeltaAIC",
    "DeltaBIC",
    "DeltaRMSE",
}

SESSION_REQUIRED_COLUMNS = {
    "Animal",
    "Protocol",
    "SessionNumber",
    "NTrials",
}

BLUE = "#3274A1"
ORANGE = "#E1812C"
INK = "#252525"
GREY = "#8A8A8A"
LIGHT_GREY = "#E6E6E6"


def _require_columns(frame, required, source_name):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {', '.join(missing)}"
        )


def _nullable_boolean(series, column_name):
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    ).astype("boolean")
    invalid = series.notna() & mapped.isna()
    if invalid.any():
        values = sorted(series.loc[invalid].astype(str).unique())
        raise ValueError(
            f"{column_name} contains invalid Boolean values: {values}"
        )
    return mapped


def attach_max_session_length(comparison, session_summary):
    """Attach an exact, reconciled longest-session bound when available."""

    comparison = comparison.copy()
    session_summary = session_summary.copy()
    _require_columns(
        comparison,
        COMPARISON_REQUIRED_COLUMNS,
        "comparison CSV",
    )
    _require_columns(
        session_summary,
        SESSION_REQUIRED_COLUMNS,
        "session summary CSV",
    )

    comparison["Animal"] = comparison["Animal"].astype("string")
    session_summary["Animal"] = session_summary["Animal"].astype("string")
    comparison["Protocol"] = comparison["Protocol"].astype("string")
    session_summary["Protocol"] = session_summary["Protocol"].astype("string")

    duplicate_comparisons = comparison.duplicated(
        ["Animal", "Protocol"],
        keep=False,
    )
    if duplicate_comparisons.any():
        keys = (
            comparison.loc[duplicate_comparisons, ["Animal", "Protocol"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"comparison CSV has duplicate Animal x Protocol rows: {keys}"
        )

    duplicate_sessions = session_summary.duplicated(
        ["Animal", "Protocol", "SessionNumber"],
        keep=False,
    )
    if duplicate_sessions.any():
        keys = (
            session_summary.loc[
                duplicate_sessions,
                ["Animal", "Protocol", "SessionNumber"],
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"session summary has duplicate session rows: {keys}")

    session_summary["NTrials"] = pd.to_numeric(
        session_summary["NTrials"],
        errors="coerce",
    )
    invalid_session_trials = (
        session_summary["NTrials"].isna()
        | (session_summary["NTrials"] <= 0)
    )
    if invalid_session_trials.any():
        raise ValueError(
            "session summary NTrials must be finite and positive for every row"
        )

    aggregated = (
        session_summary.groupby(
            ["Animal", "Protocol"],
            dropna=False,
            as_index=False,
        )
        .agg(
            _SessionCount=("SessionNumber", "size"),
            _SessionTrialSum=("NTrials", "sum"),
            _MaxSessionLength=("NTrials", "max"),
        )
    )

    original_rows = len(comparison)
    merged = comparison.merge(
        aggregated,
        on=["Animal", "Protocol"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != original_rows:
        raise RuntimeError("session metadata join changed comparison row count")

    expected_trials = pd.to_numeric(merged["NTrials"], errors="coerce")
    expected_sessions = pd.to_numeric(merged["NSessions"], errors="coerce")
    has_metadata = merged["_SessionCount"].notna()
    verified = (
        has_metadata
        & np.isclose(
            merged["_SessionTrialSum"],
            expected_trials,
            rtol=0,
            atol=0,
        )
        & np.isclose(
            merged["_SessionCount"],
            expected_sessions,
            rtol=0,
            atol=0,
        )
    )

    merged["MaxSessionLength"] = merged["_MaxSessionLength"].where(
        verified,
        np.nan,
    )
    merged["MaxSessionLengthVerified"] = pd.Series(
        verified,
        index=merged.index,
        dtype="boolean",
    )
    merged["MaxSessionLengthSource"] = pd.Series(
        np.where(
            verified,
            "exact_session_summary",
            np.where(has_metadata, "metadata_mismatch", "unavailable"),
        ),
        index=merged.index,
        dtype="string",
    )
    return merged.drop(
        columns=[
            "_SessionCount",
            "_SessionTrialSum",
            "_MaxSessionLength",
        ]
    )


def _new_nullable_boolean(index):
    return pd.Series(pd.NA, index=index, dtype="boolean")


def _set_flag(frame, name, known, condition):
    flag = _new_nullable_boolean(frame.index)
    flag.loc[known] = condition.loc[known].astype(bool)
    frame[name] = flag


def _composite_boundary_state(frame, flag_names, eligible):
    flags = frame[flag_names]
    any_true = flags.eq(True).any(axis=1)
    all_known = flags.notna().all(axis=1)
    result = _new_nullable_boolean(frame.index)
    result.loc[eligible & any_true] = False
    result.loc[eligible & all_known & ~any_true] = True
    return result


def _any_boundary_state(frame, flag_names, eligible):
    flags = frame[flag_names]
    any_true = flags.eq(True).any(axis=1)
    all_known = flags.notna().all(axis=1)
    result = _new_nullable_boolean(frame.index)
    result.loc[eligible & any_true] = True
    result.loc[eligible & all_known & ~any_true] = False
    return result


def _both_well_identified(continuous, session_aware, eligible):
    result = _new_nullable_boolean(continuous.index)
    definitely_false = continuous.eq(False) | session_aware.eq(False)
    definitely_true = continuous.eq(True) & session_aware.eq(True)
    result.loc[eligible & definitely_false.fillna(False)] = False
    result.loc[eligible & definitely_true.fillna(False)] = True
    return result


def append_qc_flags(comparison):
    """Append nullable parameter-boundary and evidence classifications."""

    frame = comparison.copy()
    _require_columns(frame, COMPARISON_REQUIRED_COLUMNS, "comparison data")

    frame["success_cont"] = _nullable_boolean(
        frame["success_cont"],
        "success_cont",
    )
    frame["success_session"] = _nullable_boolean(
        frame["success_session"],
        "success_session",
    )
    frame["BothModelsSuccessful"] = (
        frame["success_cont"] & frame["success_session"]
    ).astype("boolean")
    eligible = frame["BothModelsSuccessful"].fillna(False)

    numeric_columns = [
        "NTrials",
        "tau_cont",
        "k_cont",
        "tau_session",
        "k_session",
        "reset_A",
        "recovery_lambda",
        "DeltaBIC",
    ]
    if "MaxSessionLength" not in frame:
        frame["MaxSessionLength"] = np.nan
    if "MaxSessionLengthVerified" not in frame:
        frame["MaxSessionLengthVerified"] = _new_nullable_boolean(frame.index)
    numeric_columns.append("MaxSessionLength")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid_trials = eligible & frame["NTrials"].notna() & (frame["NTrials"] >= 1)
    first_trial = 1.0
    last_trial = frame["NTrials"]
    trial_range = last_trial - first_trial
    tau_lower = first_trial + 0.02 * trial_range
    tau_upper = last_trial - 0.02 * trial_range

    for prefix, tau_column, k_column in (
        ("Cont", "tau_cont", "k_cont"),
        ("Session", "tau_session", "k_session"),
    ):
        tau_known = valid_trials & frame[tau_column].notna()
        _set_flag(
            frame,
            f"Tau{prefix}NearLowerBound",
            tau_known,
            frame[tau_column] <= tau_lower,
        )
        _set_flag(
            frame,
            f"Tau{prefix}NearUpperBound",
            tau_known,
            frame[tau_column] >= tau_upper,
        )
        k_known = eligible & frame[k_column].notna()
        _set_flag(
            frame,
            f"K{prefix}NearLowerBound",
            k_known,
            frame[k_column] <= 1e-7,
        )
        _set_flag(
            frame,
            f"K{prefix}NearUpperBound",
            k_known,
            frame[k_column] >= 0.99,
        )

    reset_known = eligible & frame["reset_A"].notna()
    _set_flag(
        frame,
        "ResetANearLowerBound",
        reset_known,
        frame["reset_A"] <= 0.01,
    )
    _set_flag(
        frame,
        "ResetANearUpperBound",
        reset_known,
        frame["reset_A"] >= 0.99,
    )

    lambda_known = eligible & frame["recovery_lambda"].notna()
    _set_flag(
        frame,
        "RecoveryLambdaNearLowerBound",
        lambda_known,
        frame["recovery_lambda"] <= 1.01,
    )
    lambda_upper_known = (
        lambda_known
        & frame["MaxSessionLength"].notna()
        & frame["MaxSessionLengthVerified"].fillna(False)
    )
    _set_flag(
        frame,
        "RecoveryLambdaNearUpperBound",
        lambda_upper_known,
        frame["recovery_lambda"] >= 0.99 * frame["MaxSessionLength"],
    )

    frame["WellIdentifiedContinuous"] = _composite_boundary_state(
        frame,
        CONTINUOUS_FLAGS,
        eligible,
    )
    frame["WellIdentifiedSessionAware"] = _composite_boundary_state(
        frame,
        SESSION_AWARE_FLAGS,
        eligible,
    )
    frame["BothWellIdentified"] = _both_well_identified(
        frame["WellIdentifiedContinuous"],
        frame["WellIdentifiedSessionAware"],
        eligible,
    )
    frame["AnyBoundaryHit"] = _any_boundary_state(
        frame,
        ALL_BOUNDARY_FLAGS,
        eligible,
    )
    frame["SessionAwareAnyBoundaryHit"] = _any_boundary_state(
        frame,
        SESSION_AWARE_FLAGS,
        eligible,
    )

    all_known = frame[ALL_BOUNDARY_FLAGS].notna().all(axis=1)
    frame["BoundaryHitCount"] = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Int64",
    )
    frame.loc[eligible & all_known, "BoundaryHitCount"] = (
        frame.loc[eligible & all_known, ALL_BOUNDARY_FLAGS]
        .astype(int)
        .sum(axis=1)
        .astype("Int64")
    )

    evidence = pd.Series(pd.NA, index=frame.index, dtype="string")
    delta = frame["DeltaBIC"]
    evidence_known = eligible & delta.notna()
    evidence.loc[evidence_known & (delta <= -10)] = (
        "StrongSessionAwareSupport"
    )
    evidence.loc[evidence_known & (delta > -10) & (delta <= -2)] = (
        "ModerateSessionAwareSupport"
    )
    evidence.loc[evidence_known & (delta > -2) & (delta < 2)] = (
        "Inconclusive"
    )
    evidence.loc[evidence_known & (delta >= 2) & (delta < 10)] = (
        "ModerateContinuousSupport"
    )
    evidence.loc[evidence_known & (delta >= 10)] = (
        "StrongContinuousSupport"
    )
    frame["EvidenceStrength"] = evidence
    return frame


def _fraction_true(values):
    known = values.dropna().astype(bool)
    return float(known.mean()) if len(known) else np.nan


def build_qc_group_summary(qc):
    rows = []
    for (protocol, genotype), group in qc.groupby(
        ["Protocol", "Genotype"],
        dropna=False,
        sort=True,
    ):
        successful = group[group["BothModelsSuccessful"].eq(True)]
        both_well = successful["BothWellIdentified"]
        delta_bic = pd.to_numeric(successful["DeltaBIC"], errors="coerce").dropna()
        row = {
            "Protocol": protocol,
            "Genotype": genotype,
            "NBothSuccessful": len(successful),
            "NKnownBothWellIdentified": int(both_well.notna().sum()),
            "NBothWellIdentified": int(both_well.eq(True).sum()),
            "FractionBothWellIdentified": _fraction_true(both_well),
            "NKnownDeltaBIC": len(delta_bic),
            "FractionStrongSessionAwareSupport": (
                float((delta_bic <= -10).mean()) if len(delta_bic) else np.nan
            ),
            "FractionAnySessionAwareSupport": (
                float((delta_bic < -2).mean()) if len(delta_bic) else np.nan
            ),
            "FractionInconclusive": (
                float(((delta_bic > -2) & (delta_bic < 2)).mean())
                if len(delta_bic)
                else np.nan
            ),
            "FractionFavoringContinuous": (
                float((delta_bic >= 2).mean()) if len(delta_bic) else np.nan
            ),
        }
        for flag in ALL_BOUNDARY_FLAGS:
            row[f"NKnown{flag}"] = int(successful[flag].notna().sum())
            row[f"Fraction{flag}"] = _fraction_true(successful[flag])
        rows.append(row)
    return pd.DataFrame(rows)


def _iqr(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def _median(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def build_robust_parameter_summary(qc):
    successful = qc[qc["BothModelsSuccessful"].eq(True)]
    keys = (
        qc[["Protocol", "Genotype"]]
        .drop_duplicates()
        .sort_values(["Protocol", "Genotype"])
    )
    rows = []
    for key in keys.itertuples(index=False):
        group = successful[
            (successful["Protocol"] == key.Protocol)
            & (successful["Genotype"] == key.Genotype)
        ]
        subsets = (
            ("AllBothSuccessful", group),
            (
                "BothWellIdentified",
                group[group["BothWellIdentified"].eq(True)],
            ),
        )
        for subset_name, subset in subsets:
            row = {
                "Subset": subset_name,
                "Protocol": key.Protocol,
                "Genotype": key.Genotype,
                "N": len(subset),
                "MedianDeltaTau": _median(subset["DeltaTau"]),
                "IQRDeltaTau": _iqr(subset["DeltaTau"]),
                "MeanDeltaTau": _mean(subset["DeltaTau"]),
                "MedianDeltaK": _median(subset["DeltaK"]),
                "IQRDeltaK": _iqr(subset["DeltaK"]),
                "MeanDeltaK": _mean(subset["DeltaK"]),
                "MedianDeltaAIC": _median(subset["DeltaAIC"]),
                "MedianDeltaBIC": _median(subset["DeltaBIC"]),
                "MedianDeltaRMSE": _median(subset["DeltaRMSE"]),
                "MedianResetA": _median(subset["reset_A"]),
                "MedianRecoveryLambda": _median(
                    subset["recovery_lambda"]
                ),
                "FractionDeltaAICBelow0": np.nan,
                "FractionDeltaBICBelow0": np.nan,
                "FractionDeltaBICAtMostMinus10": np.nan,
                "FractionDeltaRMSEBelow0": np.nan,
            }
            if subset_name == "BothWellIdentified" and len(subset):
                row["FractionDeltaAICBelow0"] = float(
                    (subset["DeltaAIC"] < 0).mean()
                )
                row["FractionDeltaBICBelow0"] = float(
                    (subset["DeltaBIC"] < 0).mean()
                )
                row["FractionDeltaBICAtMostMinus10"] = float(
                    (subset["DeltaBIC"] <= -10).mean()
                )
                row["FractionDeltaRMSEBelow0"] = float(
                    (subset["DeltaRMSE"] < 0).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def strong_support_boundary_hits(qc):
    mask = (
        qc["BothModelsSuccessful"].eq(True)
        & (pd.to_numeric(qc["DeltaBIC"], errors="coerce") <= -10)
        & qc["SessionAwareAnyBoundaryHit"].eq(True)
    )
    return qc.loc[mask].copy()


def _clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def plot_parameter_boundary_hits(group_summary, output_dir):
    labels = [
        "tau cont low",
        "tau cont high",
        "k cont low",
        "k cont high",
        "tau session low",
        "tau session high",
        "k session low",
        "k session high",
        "A low",
        "A high",
        "lambda low",
        "lambda high",
    ]
    columns = [f"Fraction{flag}" for flag in ALL_BOUNDARY_FLAGS]
    row_labels = [
        f"{row.Protocol} - {row.Genotype} (n={row.NBothSuccessful})"
        for row in group_summary.itertuples(index=False)
    ]
    matrix = group_summary[columns].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    fig_width = max(12, 0.85 * len(columns))
    fig_height = max(4, 0.65 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Parameter-boundary hit fractions")
    ax.set_xlabel("QC boundary warning")
    ax.set_ylabel("Protocol - genotype; denominator is both-successful fits")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value >= 0.55 else INK
            ax.text(column, row, text, ha="center", va="center", color=color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Fraction hitting bound")
    fig.tight_layout()
    fig.savefig(
        output_dir / "parameter_boundary_hits.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def _category_order(frame):
    protocols = [value for value in ("AB", "CD") if value in set(frame["Protocol"])]
    genotypes = [value for value in ("WT", "HET") if value in set(frame["Genotype"])]
    remaining_protocols = sorted(set(frame["Protocol"]) - set(protocols))
    remaining_genotypes = sorted(set(frame["Genotype"]) - set(genotypes))
    return [
        (protocol, genotype)
        for protocol in protocols + remaining_protocols
        for genotype in genotypes + remaining_genotypes
        if ((frame["Protocol"] == protocol) & (frame["Genotype"] == genotype)).any()
    ]


def plot_robust_delta(qc, metric, output_path):
    frame = qc[qc["BothModelsSuccessful"].eq(True)].copy()
    categories = _category_order(frame)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    styles = {
        "Well identified": {"marker": "o", "color": BLUE, "facecolors": BLUE},
        "Boundary warning": {"marker": "x", "color": ORANGE},
        "QC unknown": {"marker": "D", "color": GREY, "facecolors": "none"},
    }

    for position, (protocol, genotype) in enumerate(categories):
        group = frame[
            (frame["Protocol"] == protocol)
            & (frame["Genotype"] == genotype)
        ]
        status_masks = {
            "Well identified": group["BothWellIdentified"].eq(True),
            "Boundary warning": group["AnyBoundaryHit"].eq(True),
            "QC unknown": group["BothWellIdentified"].isna(),
        }
        for status, mask in status_masks.items():
            values = pd.to_numeric(group.loc[mask, metric], errors="coerce").dropna()
            if values.empty:
                continue
            jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.array([0.0])
            style = styles[status]
            kwargs = {
                "marker": style["marker"],
                "color": style["color"],
                "s": 45,
                "alpha": 0.85,
                "label": status if position == 0 else None,
            }
            if "facecolors" in style:
                kwargs["facecolors"] = style["facecolors"]
            ax.scatter(position + jitter, values, **kwargs)

        well_values = pd.to_numeric(
            group.loc[group["BothWellIdentified"].eq(True), metric],
            errors="coerce",
        ).dropna()
        if len(well_values):
            median = float(well_values.median())
            ax.plot(
                [position - 0.22, position + 0.22],
                [median, median],
                color=INK,
                linewidth=2.5,
            )

    ax.axhline(0, color=INK, linestyle="--", linewidth=1)
    ax.set_xticks(
        range(len(categories)),
        [f"{protocol}\n{genotype}" for protocol, genotype in categories],
    )
    ax.set_ylabel(f"{metric} (session-aware minus continuous)")
    ax.set_title(f"Robust {metric} comparison", pad=28)
    ax.text(
        0.01,
        1.02,
        "Individual animals; black segment is the well-identified median",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    _clean_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bic_vs_qc(qc, output_dir):
    frame = qc[qc["BothModelsSuccessful"].eq(True)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    statuses = ["Well identified", "Boundary warning", "QC unknown"]
    genotype_styles = {
        "WT": {"color": BLUE, "marker": "o"},
        "HET": {"color": ORANGE, "marker": "s"},
    }

    for ax, protocol in zip(axes, ("AB", "CD")):
        protocol_frame = frame[frame["Protocol"] == protocol]
        for status_position, status in enumerate(statuses):
            if status == "Well identified":
                status_mask = protocol_frame["BothWellIdentified"].eq(True)
            elif status == "Boundary warning":
                status_mask = protocol_frame["AnyBoundaryHit"].eq(True)
            else:
                status_mask = protocol_frame["BothWellIdentified"].isna()

            for genotype in sorted(protocol_frame["Genotype"].dropna().unique()):
                values = pd.to_numeric(
                    protocol_frame.loc[
                        status_mask & (protocol_frame["Genotype"] == genotype),
                        "DeltaBIC",
                    ],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                jitter = np.linspace(-0.1, 0.1, len(values)) if len(values) > 1 else np.array([0.0])
                style = genotype_styles.get(
                    str(genotype),
                    {"color": GREY, "marker": "D"},
                )
                ax.scatter(
                    status_position + jitter,
                    values,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.8,
                    s=45,
                    label=str(genotype) if status_position == 0 else None,
                )

        for reference, linestyle in ((-10, "--"), (-2, ":"), (2, ":"), (10, "--")):
            ax.axhline(reference, color=GREY, linestyle=linestyle, linewidth=0.9)
        ax.set_xticks(range(len(statuses)), statuses, rotation=20, ha="right")
        ax.set_title(protocol)
        ax.set_xlabel("Parameter QC status")
        _clean_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), frameon=False)

    axes[0].set_ylabel("DeltaBIC (session-aware minus continuous)")
    fig.suptitle("DeltaBIC by parameter-QC status")
    fig.tight_layout()
    fig.savefig(
        output_dir / "delta_bic_vs_qc.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def _readme_text(comparison_path, session_summary_path):
    flag_lines = "\n".join(
        [
            "- TauContNearLowerBound: tau_cont <= 1 + 0.02*(NTrials-1)",
            "- TauContNearUpperBound: tau_cont >= NTrials - 0.02*(NTrials-1)",
            "- KContNearLowerBound: k_cont <= 1e-7",
            "- KContNearUpperBound: k_cont >= 0.99",
            "- TauSessionNearLowerBound: tau_session <= 1 + 0.02*(NTrials-1)",
            "- TauSessionNearUpperBound: tau_session >= NTrials - 0.02*(NTrials-1)",
            "- KSessionNearLowerBound: k_session <= 1e-7",
            "- KSessionNearUpperBound: k_session >= 0.99",
            "- ResetANearLowerBound: reset_A <= 0.01",
            "- ResetANearUpperBound: reset_A >= 0.99",
            "- RecoveryLambdaNearLowerBound: recovery_lambda <= 1.01",
            "- RecoveryLambdaNearUpperBound: recovery_lambda >= 0.99*MaxSessionLength",
        ]
    )
    return f"""Session-aware learning model QC

Purpose and inputs
This diagnostic reads existing model-comparison results and does not rerun
behavioral preprocessing or refit either model.
Comparison source: {comparison_path}
Session metadata source: {session_summary_path}

Exact lambda upper bound
MaxSessionLength is derived by grouping the session summary at Animal x
Protocol and taking the maximum retained session NTrials. It is accepted only
when the session count equals NSessions and summed session NTrials equals the
comparison row's NTrials. MaxSessionLengthVerified and
MaxSessionLengthSource record this check. No approximate bound is guessed.

Eligibility and three-state logic
Parameter QC is calculated only when both optimizers succeeded. QC flags use
nullable Boolean values: True is a boundary warning, False is a verified
non-hit, and NA means the flag could not be evaluated. A model is well
identified only when every required flag is known and False. Any known True
flag makes the corresponding well-identified classification False; otherwise
an unknown constituent keeps the composite NA. Group fractions exclude NA
from their denominators and report the corresponding NKnown columns.

Boundary-warning thresholds
The fitted tau range is 1 through NTrials, with trial_range=NTrials-1.
{flag_lines}

Composite fields
WellIdentifiedContinuous requires all four continuous tau/k flags to be
known and False. WellIdentifiedSessionAware requires all eight session-aware
tau/k/A/lambda flags to be known and False. BothWellIdentified requires both
model classifications to be True. AnyBoundaryHit and
SessionAwareAnyBoundaryHit distinguish all-family from session-aware-only
warnings. BoundaryHitCount is NA unless all twelve flags are known.

DeltaBIC evidence classes
- StrongSessionAwareSupport: DeltaBIC <= -10
- ModerateSessionAwareSupport: -10 < DeltaBIC <= -2
- Inconclusive: -2 < DeltaBIC < 2
- ModerateContinuousSupport: 2 <= DeltaBIC < 10
- StrongContinuousSupport: DeltaBIC >= 10
Negative DeltaBIC favors the session-aware model.

Robust interpretation
Boundary hits are warnings, not automatic invalidation of a model family or
its fit advantage. Mean DeltaTau and DeltaK can be dominated by pathological
or boundary-constrained fits. Median/IQR summaries and the BothWellIdentified
subset are the primary robust summaries. strong_support_but_boundary_hit.csv
specifically separates strong model-family support from trustworthy
session-aware parameter estimation.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--session-summary", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    comparison_path = Path(args.comparison_csv).expanduser().resolve()
    session_summary_path = Path(args.session_summary).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else comparison_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(comparison_path, dtype={"Animal": "string"})
    session_summary = pd.read_csv(
        session_summary_path,
        dtype={"Animal": "string"},
    )
    qc = append_qc_flags(
        attach_max_session_length(comparison, session_summary)
    )
    group_summary = build_qc_group_summary(qc)
    robust_summary = build_robust_parameter_summary(qc)
    strong_boundary = strong_support_boundary_hits(qc)

    qc.to_csv(output_dir / "session_aware_model_qc.csv", index=False)
    group_summary.to_csv(
        output_dir / "session_aware_model_qc_group_summary.csv",
        index=False,
    )
    robust_summary.to_csv(
        output_dir / "robust_parameter_change_summary.csv",
        index=False,
    )
    strong_boundary.to_csv(
        output_dir / "strong_support_but_boundary_hit.csv",
        index=False,
    )

    plot_parameter_boundary_hits(group_summary, output_dir)
    plot_robust_delta(
        qc,
        "DeltaTau",
        output_dir / "robust_delta_tau.png",
    )
    plot_robust_delta(
        qc,
        "DeltaK",
        output_dir / "robust_delta_k.png",
    )
    plot_delta_bic_vs_qc(qc, output_dir)
    (output_dir / "README_session_aware_model_qc.txt").write_text(
        _readme_text(comparison_path, session_summary_path),
        encoding="utf-8",
    )

    successful = qc[qc["BothModelsSuccessful"].eq(True)]
    well = successful["BothWellIdentified"]
    print(f"Session-aware model QC complete: {output_dir}")
    print(f"Rows: {len(qc)}")
    print(f"Both-successful rows: {len(successful)}")
    print(
        "Verified max-session bounds among both-successful rows: "
        f"{int(successful['MaxSessionLengthVerified'].eq(True).sum())}"
    )
    print(
        "Both well identified: "
        f"{int(well.eq(True).sum())}/{int(well.notna().sum())}"
    )
    print(f"Strong support with session-aware boundary hit: {len(strong_boundary)}")


if __name__ == "__main__":
    main()
