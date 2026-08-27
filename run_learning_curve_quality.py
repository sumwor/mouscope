"""Run the existing sigmoid learning-curve model and visualize fit quality.

This is an external analysis wrapper.  It does not modify the implementation in
``BehDataOdor.find_eureka`` and it never writes to the source data tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import mannwhitneyu

from behavioral_pipeline import BehDataOdor


METRICS = {
    "FittedTau": "Fitted tau (trial)",
    "PlateauPerformance": "Plateau performance",
    "FittedK": "Fitted K",
}
COLORS = {"WT": "#4C78A8", "HET": "#E45756", "KO": "#72B7B2"}


# Matplotlib 3.9 renamed the boxplot keyword from ``labels`` to
# ``tick_labels``.  Keep the repository's unchanged plotting code runnable on
# both sides of that API change.
_original_boxplot = Axes.boxplot


def _compatible_boxplot(self, *args, **kwargs):
    if "labels" in kwargs and "tick_labels" not in kwargs:
        kwargs["tick_labels"] = kwargs.pop("labels")
    return _original_boxplot(self, *args, **kwargs)


Axes.boxplot = _compatible_boxplot


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


def concatenate_behavior(model: BehDataOdor, animal: str) -> dict[str, pd.DataFrame]:
    result = {"AB": pd.DataFrame(), "CD": pd.DataFrame()}
    sessions = model.data_index[model.data_index["Animal"] == animal]
    transition = sessions[sessions["Protocol"].str.contains("AB-CD")]
    first_transition_idx = transition.index[0] if not transition.empty else None
    ab_sessions = sessions.loc[: first_transition_idx - 1] if first_transition_idx is not None else sessions
    cd_sessions = sessions[
        (sessions["Protocol"] == "AB-CD") & (sessions["ProtocolDay"] <= 3)
    ]

    for session_idx in ab_sessions.index:
        frame = pd.read_csv(ab_sessions.loc[session_idx, "BehCSV"])
        frame = frame[~np.isnan(frame["actions"])]
        result["AB"] = pd.concat([result["AB"], frame], ignore_index=True)

    for session_idx in cd_sessions.index:
        frame = pd.read_csv(cd_sessions.loc[session_idx, "BehCSV"])
        frame = frame[~np.isnan(frame["actions"])]
        frame = frame[frame["schedule"] > 2]
        result["CD"] = pd.concat([result["CD"], frame], ignore_index=True)

    return result


def observed_learning_curve(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    reward = frame["reward"].fillna(0).replace([2, 3], 1)
    rewarded = (reward > 0).astype(float).to_numpy()
    n_trials = len(rewarded)
    window_size = 60
    running = np.full(n_trials, np.nan)
    if n_trials >= window_size:
        csum = np.empty(n_trials + 1, dtype=float)
        csum[0] = 0.0
        np.cumsum(rewarded, out=csum[1:])
        running[: n_trials - window_size + 1] = (
            csum[window_size:] - csum[:-window_size]
        ) / window_size
    smoothed = (
        pd.Series(running)
        .rolling(500, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    valid = np.flatnonzero(np.isfinite(smoothed))
    return valid.astype(float) + 1, smoothed[valid]


def sigmoid_prediction(params, x: np.ndarray) -> np.ndarray:
    p_low, p_delta, k, tau = (float(value) for value in params)
    exponent = np.clip(-k * (x - tau), -700, 700)
    return p_low + p_delta / (1 + np.exp(exponent))


def fit_window_mask(saved: dict, x: np.ndarray) -> np.ndarray:
    learning_window = saved.get("learning_window")
    plateau_window = saved.get("plateau_window")
    if not learning_window or not plateau_window:
        return np.ones(len(x), dtype=bool)
    learning_start, learning_end = (float(value) for value in learning_window)
    plateau_start, plateau_end = (float(value) for value in plateau_window)
    fit_pad = max(250.0, learning_end - learning_start)
    fit_start = max(1.0, learning_start - fit_pad)
    fit_end = min(plateau_start + 400.0, plateau_end)
    return (x >= fit_start) & (x <= fit_end)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "n_points": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
    }


def collect_fit_quality(model: BehDataOdor) -> tuple[pd.DataFrame, dict]:
    rows = []
    curves = {}
    animal_info = model.data_index.drop_duplicates("Animal").set_index("Animal")
    for animal in model.data_index["Animal"].unique():
        concatenated = concatenate_behavior(model, animal)
        curves[animal] = {}
        for protocol in ("AB", "CD"):
            x, observed = observed_learning_curve(concatenated[protocol])
            saved = model.eureka_learning[animal][protocol]
            params = saved.get("sigmoid_params")
            base = {
                "Animal": animal,
                "Genotype": animal_info.loc[animal, "Genotype"],
                "Gender": animal_info.loc[animal, "Gender"],
                "Protocol": protocol,
                "n_curve_points": len(x),
                "fit_available": params is not None,
            }
            if params is None or len(x) == 0:
                rows.append(base)
                continue
            predicted = sigmoid_prediction(params, x)
            mask = fit_window_mask(saved, x)
            full = regression_metrics(observed, predicted)
            fit = regression_metrics(observed[mask], predicted[mask])
            rows.append(
                {
                    **base,
                    "r2_full_curve": full["r2"],
                    "rmse_full_curve": full["rmse"],
                    "mae_full_curve": full["mae"],
                    "n_fit_points": fit["n_points"],
                    "r2_fit_window": fit["r2"],
                    "rmse_fit_window": fit["rmse"],
                    "mae_fit_window": fit["mae"],
                }
            )
            curves[animal][protocol] = (x, observed, predicted, mask)
    return pd.DataFrame(rows), curves


def bh_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return adjusted
    n_tests = len(valid)
    ranked = valid.to_numpy() * n_tests / np.arange(1, n_tests + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted.loc[valid.index] = np.minimum(ranked, 1.0)
    return adjusted


def summarize_parameters(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    comparison_rows = []
    for protocol in summary["Protocol"].dropna().unique():
        protocol_df = summary[summary["Protocol"] == protocol]
        genotypes = list(protocol_df["Genotype"].dropna().unique())
        for metric in METRICS:
            for genotype in genotypes:
                values = pd.to_numeric(
                    protocol_df.loc[protocol_df["Genotype"] == genotype, metric],
                    errors="coerce",
                )
                valid = values.dropna()
                summary_rows.append(
                    {
                        "Protocol": protocol,
                        "Genotype": genotype,
                        "Metric": metric,
                        "N_total": len(values),
                        "N_valid": len(valid),
                        "N_missing": int(values.isna().sum()),
                        "Mean": valid.mean(),
                        "SD": valid.std(ddof=1),
                        "SEM": valid.sem(ddof=1),
                        "Median": valid.median(),
                        "Q1": valid.quantile(0.25),
                        "Q3": valid.quantile(0.75),
                        "Min": valid.min(),
                        "Max": valid.max(),
                    }
                )
            for idx, genotype_a in enumerate(genotypes):
                values_a = pd.to_numeric(
                    protocol_df.loc[protocol_df["Genotype"] == genotype_a, metric],
                    errors="coerce",
                ).dropna()
                for genotype_b in genotypes[idx + 1 :]:
                    values_b = pd.to_numeric(
                        protocol_df.loc[protocol_df["Genotype"] == genotype_b, metric],
                        errors="coerce",
                    ).dropna()
                    if len(values_a) and len(values_b):
                        statistic, p_value = mannwhitneyu(
                            values_a, values_b, alternative="two-sided"
                        )
                    else:
                        statistic, p_value = np.nan, np.nan
                    comparison_rows.append(
                        {
                            "Protocol": protocol,
                            "Metric": metric,
                            "GenotypeA": genotype_a,
                            "GenotypeB": genotype_b,
                            "N_A": len(values_a),
                            "N_B": len(values_b),
                            "MannWhitneyU": statistic,
                            "PValue": p_value,
                        }
                    )
    descriptive = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["FDR_BH_PValue"] = bh_adjust(comparisons["PValue"])
    comparisons["FDR_Significant_0.05"] = comparisons["FDR_BH_PValue"] < 0.05
    return descriptive, comparisons


def ordered_groups(frame: pd.DataFrame) -> list[tuple[str, str]]:
    protocol_order = [p for p in ("AB", "CD") if p in set(frame["Protocol"])]
    protocol_order += [p for p in frame["Protocol"].unique() if p not in protocol_order]
    genotype_order = [g for g in ("WT", "HET", "KO") if g in set(frame["Genotype"])]
    genotype_order += [g for g in frame["Genotype"].unique() if g not in genotype_order]
    return [(protocol, genotype) for protocol in protocol_order for genotype in genotype_order]


def save_figure(fig, output_dir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_box_scatter(summary: pd.DataFrame, output_dir: Path):
    groups = ordered_groups(summary)
    labels = [f"{protocol}\n{genotype}" for protocol, genotype in groups]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, ylabel) in zip(axes, METRICS.items()):
        values_by_group = []
        for protocol, genotype in groups:
            values = pd.to_numeric(
                summary.loc[
                    (summary["Protocol"] == protocol)
                    & (summary["Genotype"] == genotype),
                    metric,
                ],
                errors="coerce",
            ).dropna().to_numpy()
            values_by_group.append(values)
        box = ax.boxplot(values_by_group, labels=labels, patch_artist=True, showfliers=False)
        for patch, (_, genotype) in zip(box["boxes"], groups):
            patch.set_facecolor(COLORS.get(genotype, "#B8B8B8"))
            patch.set_alpha(0.35)
        for position, (values, (_, genotype)) in enumerate(
            zip(values_by_group, groups), start=1
        ):
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=30,
                color=COLORS.get(genotype, "#555555"),
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Sigmoid learning-fit parameters by protocol and genotype")
    save_figure(fig, output_dir, "learning_parameter_box_scatter")


def plot_mean_sem(summary: pd.DataFrame, output_dir: Path):
    protocols = [p for p in ("AB", "CD") if p in set(summary["Protocol"])]
    genotypes = [g for g in ("WT", "HET", "KO") if g in set(summary["Genotype"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(protocols))
    for ax, (metric, ylabel) in zip(axes, METRICS.items()):
        for genotype in genotypes:
            means = []
            sems = []
            for protocol in protocols:
                values = pd.to_numeric(
                    summary.loc[
                        (summary["Protocol"] == protocol)
                        & (summary["Genotype"] == genotype),
                        metric,
                    ],
                    errors="coerce",
                ).dropna()
                means.append(values.mean())
                sems.append(values.sem(ddof=1))
            ax.errorbar(
                x,
                means,
                yerr=sems,
                marker="o",
                capsize=4,
                linewidth=2,
                label=genotype,
                color=COLORS.get(genotype, "#555555"),
            )
        ax.set_xticks(x, protocols)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].legend(frameon=False)
    fig.suptitle("Mean ± SEM of sigmoid learning-fit parameters")
    save_figure(fig, output_dir, "learning_parameter_mean_sem")


def plot_quality(quality: pd.DataFrame, output_dir: Path):
    available = quality[quality["fit_available"]].copy()
    groups = ordered_groups(available)
    labels = [f"{protocol}\n{genotype}" for protocol, genotype in groups]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    quality_metrics = {
        "r2_fit_window": "R² (fit window)",
        "rmse_fit_window": "RMSE (fit window)",
        "mae_fit_window": "MAE (fit window)",
    }
    for ax, (metric, ylabel) in zip(axes, quality_metrics.items()):
        values_by_group = [
            pd.to_numeric(
                available.loc[
                    (available["Protocol"] == protocol)
                    & (available["Genotype"] == genotype),
                    metric,
                ],
                errors="coerce",
            ).dropna().to_numpy()
            for protocol, genotype in groups
        ]
        box = ax.boxplot(values_by_group, labels=labels, patch_artist=True, showfliers=False)
        for patch, (_, genotype) in zip(box["boxes"], groups):
            patch.set_facecolor(COLORS.get(genotype, "#B8B8B8"))
            patch.set_alpha(0.35)
        for position, (values, (_, genotype)) in enumerate(
            zip(values_by_group, groups), start=1
        ):
            ax.scatter(
                np.full(len(values), position),
                values,
                s=24,
                color=COLORS.get(genotype, "#555555"),
                alpha=0.75,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Single-animal sigmoid fit quality")
    save_figure(fig, output_dir, "learning_fit_quality")


def plot_quality_examples(quality: pd.DataFrame, curves: dict, output_dir: Path):
    selections = []
    available = quality.dropna(subset=["r2_fit_window"])
    for protocol in ("AB", "CD"):
        protocol_df = available[available["Protocol"] == protocol].sort_values(
            "r2_fit_window"
        )
        if protocol_df.empty:
            continue
        indices = sorted(set([0, len(protocol_df) // 2, len(protocol_df) - 1]))
        selections.extend(protocol_df.iloc[indices].to_dict("records"))
    if not selections:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ax, row in zip(axes.ravel(), selections):
        x, observed, predicted, mask = curves[row["Animal"]][row["Protocol"]]
        ax.plot(x, observed, color="black", linewidth=2, label="Observed")
        ax.plot(x, predicted, color="#E45756", linewidth=2, label="Sigmoid")
        if mask.any():
            ax.axvspan(x[mask][0], x[mask][-1], color="#F2CF5B", alpha=0.15)
        ax.set_title(
            f'{row["Animal"]} {row["Genotype"]} {row["Protocol"]}\n'
            f'R²={row["r2_fit_window"]:.3f}, RMSE={row["rmse_fit_window"]:.3f}'
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel("P(correct)")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(selections) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Low, median, and high fit-quality examples (yellow = fit window)")
    save_figure(fig, output_dir, "learning_fit_quality_examples")


def write_audit(
    source_summary: pd.DataFrame,
    regenerated_summary: pd.DataFrame,
    quality: pd.DataFrame,
    model: BehDataOdor,
    output_dir: Path,
):
    rows = [
        ("source_summary_rows", len(source_summary)),
        ("source_summary_animals", source_summary["Animal"].nunique()),
        ("regenerated_summary_rows", len(regenerated_summary)),
        ("regenerated_summary_animals", regenerated_summary["Animal"].nunique()),
        ("behavior_sessions", len(model.data_index)),
        ("behavior_animals", model.data_index["Animal"].nunique()),
        ("single_animal_fits_available", int(quality["fit_available"].sum())),
        ("single_animal_fits_unavailable", int((~quality["fit_available"]).sum())),
    ]
    for column in ("FittedTau", "PlateauPerformance", "FittedK"):
        rows.append((f"source_missing_{column}", int(source_summary[column].isna().sum())))
        rows.append(
            (f"regenerated_missing_{column}", int(regenerated_summary[column].isna().sum()))
        )
    pd.DataFrame(rows, columns=["Check", "Value"]).to_csv(
        output_dir / "learning_data_audit.csv", index=False
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-dir")
    parser.add_argument("--strain")
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.summary_csv).expanduser().resolve()
    root_dir = (
        Path(args.root_dir).expanduser().resolve()
        if args.root_dir
        else source_path.parent.parent
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strain = args.strain or root_dir.name

    source_summary = pd.read_csv(source_path)
    model = ReadOnlyBehDataOdor(str(root_dir), strain)
    model.summary = str(output_dir)
    model.find_eureka()

    regenerated_summary = model.eureka_learning_summary.copy()
    quality, curves = collect_fit_quality(model)
    descriptive, comparisons = summarize_parameters(regenerated_summary)

    descriptive.to_csv(output_dir / "learning_summary_statistics.csv", index=False)
    comparisons.to_csv(output_dir / "learning_genotype_comparisons.csv", index=False)
    quality.to_csv(output_dir / "learning_fit_quality_metrics.csv", index=False)
    write_audit(source_summary, regenerated_summary, quality, model, output_dir)

    plot_box_scatter(regenerated_summary, output_dir)
    plot_mean_sem(regenerated_summary, output_dir)
    plot_quality(quality, output_dir)
    plot_quality_examples(quality, curves, output_dir)

    note = (
        "The supplied eureka_learning_summary.csv is a fitted-parameter summary and "
        "cannot by itself assess single-animal sigmoid fit quality. This run found "
        "the processed trial-level behavior CSVs under the same TSC2_adol root, "
        "reran the unchanged BehDataOdor.find_eureka() implementation with outputs "
        "redirected here, and calculated R2/RMSE/MAE externally from the regenerated "
        "observed and predicted learning curves. Source data files were read only.\n"
    )
    (output_dir / "README_learning_analysis.txt").write_text(note, encoding="utf-8")
    print(f"Learning-curve analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
