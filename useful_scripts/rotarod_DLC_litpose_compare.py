"""Compare DeepLabCut and Lightning Pose/LitPose keypoint predictions.

The output is aimed at finding tracking failures: a keypoint that moves an
implausibly large distance between two adjacent frames.  CSV files in the DLC
and litpose folders are paired by their video name and only body parts shared
by both files are compared.

Example
-------
python rotarod_DLC_litpose_compare.py --jump-threshold 20
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()
import numpy as np
import pandas as pd


DATA_FOLDER = Path(r"Y:\HongliWang\Rotarod\DLC_litpose_comparison")
DLC_FOLDER = DATA_FOLDER / "DLC"
LITPOSE_FOLDER = DATA_FOLDER / "litpose"


def _clean_name(value: object) -> str:
    """Return a case-insensitive name suitable for matching columns/files.

    This deliberately removes spaces and punctuation, so labels such as
    ``spine 3``, ``spine_3``, and ``spine3`` all refer to the same body part.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _video_key(path: Path) -> str:
    """Remove common tracker suffixes so DLC and LitPose files can be paired."""
    name = path.stem.lower()
    # DLC commonly appends scorer/shuffle information after the video stem.
    name = re.sub(r"(?:_)?dlc.*$", "", name)
    name = re.sub(r"(?:_)?(?:litpose|lightningpose|predictions?|labels?).*$", "", name)
    # Trimmed DLC videos are often named e.g. ``trial_trimDLC...csv``;
    # ``trim`` is a processing suffix, not part of the original video name.
    name = re.sub(r"(?:[_\-\s]*trim)+$", "", name)
    return _clean_name(name)


def _as_keypoint_table_from_dlc(path: Path) -> pd.DataFrame | None:
    """Read a DLC-style three-row header into frame/bodypart x/y/likelihood."""
    try:
        table = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    except (ValueError, pd.errors.ParserError):
        return None
    if not isinstance(table.columns, pd.MultiIndex):
        return None

    coordinates = {_clean_name(v) for v in table.columns.get_level_values(-1)}
    if not {"x", "y"}.issubset(coordinates):
        return None

    rows = []
    for bodypart in table.columns.get_level_values(1).unique():
        cols = table.loc[:, table.columns.get_level_values(1) == bodypart]
        by_coord = {_clean_name(col[-1]): col for col in cols.columns}
        if not {"x", "y"}.issubset(by_coord):
            continue
        item = pd.DataFrame({
            "frame": table.index,
            "bodypart": str(bodypart),
            "x": pd.to_numeric(table[by_coord["x"]], errors="coerce"),
            "y": pd.to_numeric(table[by_coord["y"]], errors="coerce"),
        })
        likelihood_col = next((by_coord[c] for c in ("likelihood", "confidence", "score", "p") if c in by_coord), None)
        item["likelihood"] = (pd.to_numeric(table[likelihood_col], errors="coerce")
                              if likelihood_col is not None else np.nan)
        rows.append(item)
    return pd.concat(rows, ignore_index=True) if rows else None


def _as_keypoint_table_from_flat(path: Path) -> pd.DataFrame:
    """Read flattened (``nose_x``, ``nose_y``) or long-format prediction CSVs."""
    table = pd.read_csv(path)
    table = table.loc[:, ~table.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    lower = {_clean_name(column): column for column in table.columns}

    # Long format: one row per bodypart/frame, e.g. frame, keypoint, x, y.
    bodypart_col = next((lower[k] for k in ("bodypart", "keypoint", "node", "landmark", "part") if k in lower), None)
    x_col, y_col = lower.get("x"), lower.get("y")
    if bodypart_col and x_col and y_col:
        frame_col = next((lower[k] for k in ("frame", "frameindex", "index") if k in lower), None)
        result = pd.DataFrame({
            "frame": table[frame_col] if frame_col else table.groupby(bodypart_col).cumcount(),
            "bodypart": table[bodypart_col].astype(str),
            "x": pd.to_numeric(table[x_col], errors="coerce"),
            "y": pd.to_numeric(table[y_col], errors="coerce"),
        })
        likelihood_col = next((lower[k] for k in ("likelihood", "confidence", "score", "p") if k in lower), None)
        result["likelihood"] = pd.to_numeric(table[likelihood_col], errors="coerce") if likelihood_col else np.nan
        return result

    # Wide format.  Accept nose_x, nose.x, x_nose, etc.
    frame_col = next((lower[k] for k in ("frame", "frameindex", "index") if k in lower), None)
    result = []
    coord_columns: dict[str, dict[str, str]] = {}
    for column in table.columns:
        tokens = [t for t in re.split(r"[^A-Za-z0-9]+", str(column).lower()) if t]
        if len(tokens) < 2:
            continue
        if tokens[-1] in {"x", "y", "likelihood", "confidence", "score", "p"}:
            coord, bodypart = tokens[-1], "_".join(tokens[:-1])
        elif tokens[0] in {"x", "y", "likelihood", "confidence", "score", "p"}:
            coord, bodypart = tokens[0], "_".join(tokens[1:])
        else:
            continue
        coord_columns.setdefault(bodypart, {})[coord] = column
    for bodypart, columns in coord_columns.items():
        if not {"x", "y"}.issubset(columns):
            continue
        likelihood_col = next((columns[c] for c in ("likelihood", "confidence", "score", "p") if c in columns), None)
        result.append(pd.DataFrame({
            "frame": table[frame_col] if frame_col else table.index,
            "bodypart": bodypart,
            "x": pd.to_numeric(table[columns["x"]], errors="coerce"),
            "y": pd.to_numeric(table[columns["y"]], errors="coerce"),
            "likelihood": pd.to_numeric(table[likelihood_col], errors="coerce") if likelihood_col else np.nan,
        }))
    if not result:
        raise ValueError("No x/y keypoint columns were found")
    return pd.concat(result, ignore_index=True)


def read_keypoints(path: Path) -> pd.DataFrame:
    """Read either a DLC three-level CSV or a common LitPose CSV layout."""
    result = _as_keypoint_table_from_dlc(path)
    if result is None:
        result = _as_keypoint_table_from_flat(path)
    result["bodypart_key"] = result["bodypart"].map(_clean_name)
    return result.dropna(subset=["x", "y"])


def count_hind_body_opposite_side_frames(keypoints: pd.DataFrame, frame_width: float) -> pd.DataFrame:
    """Return frames with either foot on one side and the hind body on the other.

    A frame qualifies when at least one of ``left foot`` or ``right foot`` is
    left (or right) of the vertical midpoint, while ``spine 3``, ``tail 1``,
    and ``tail 2`` are all on the opposite side. The non-matching foot is
    unrestricted. Frames are excluded only when a hind-body point or both feet
    are missing.
    """
    required = ["leftfoot", "rightfoot", "spine3", "tail1", "tail2"]
    positions = (keypoints.loc[keypoints.bodypart_key.isin(required)]
                 .pivot_table(index="frame", columns="bodypart_key", values="x", aggfunc="first")
                 .reindex(columns=required))
    midpoint = frame_width / 2
    hind_left = (positions.spine3 < midpoint) & (positions.tail1 < midpoint) & (positions.tail2 < midpoint)
    hind_right = (positions.spine3 >= midpoint) & (positions.tail1 >= midpoint) & (positions.tail2 >= midpoint)
    left_foot_matches = ((positions.leftfoot < midpoint) & hind_right) | ((positions.leftfoot >= midpoint) & hind_left)
    right_foot_matches = ((positions.rightfoot < midpoint) & hind_right) | ((positions.rightfoot >= midpoint) & hind_left)
    events = positions.loc[left_foot_matches | right_foot_matches].copy()
    events["matching_foot"] = np.select(
        [left_foot_matches.loc[events.index] & right_foot_matches.loc[events.index],
         left_foot_matches.loc[events.index]],
        ["left foot; right foot", "left foot"],
        default="right foot",
    )
    events["feet_side"] = np.where(hind_left.loc[events.index], "right", "left")
    events["hind_body_side"] = np.where(hind_left.loc[events.index], "left", "right")
    return events.reset_index()[["frame", "matching_foot", "feet_side", "hind_body_side"]]


def compare_video(dlc_path: Path, litpose_path: Path, jump_threshold: float) -> pd.DataFrame:
    """Return a row for each shared keypoint/frame pair with jump statistics."""
    dlc = read_keypoints(dlc_path).rename(columns={c: f"dlc_{c}" for c in ("x", "y", "likelihood")})
    litpose = read_keypoints(litpose_path).rename(columns={c: f"litpose_{c}" for c in ("x", "y", "likelihood")})
    merged = dlc.merge(litpose, on=["frame", "bodypart_key"], how="inner", suffixes=("_dlc", "_litpose"))
    if merged.empty:
        return merged
    merged["bodypart"] = merged["bodypart_dlc"]
    merged = merged.sort_values(["bodypart_key", "frame"]).copy()
    for tracker in ("dlc", "litpose"):
        dx = merged.groupby("bodypart_key")[f"{tracker}_x"].diff()
        dy = merged.groupby("bodypart_key")[f"{tracker}_y"].diff()
        merged[f"{tracker}_jump_px"] = np.hypot(dx, dy)
        merged[f"{tracker}_is_jump"] = merged[f"{tracker}_jump_px"] >= jump_threshold
    merged["tracker_difference_px"] = np.hypot(merged.dlc_x - merged.litpose_x, merged.dlc_y - merged.litpose_y)
    merged["jump_disagreement"] = merged.dlc_is_jump != merged.litpose_is_jump
    merged["video"] = dlc_path.stem
    return merged


def save_distance_distribution(results: pd.DataFrame, output_path: Path) -> None:
    """Plot DLC and LitPose adjacent-frame displacement distributions by body part."""
    bodyparts = sorted(results["bodypart"].dropna().unique())
    ncols = min(3, len(bodyparts))
    nrows = int(np.ceil(len(bodyparts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)

    for axis, bodypart in zip(axes.flat, bodyparts):
        subset = results.loc[results.bodypart == bodypart]
        all_distances = pd.concat([
            subset["dlc_jump_px"].dropna(),
            subset["litpose_jump_px"].dropna(),
        ])
        bins = np.histogram_bin_edges(all_distances, bins="auto")
        for tracker, color in (("dlc", "#1f77b4"), ("litpose", "#ff7f0e")):
            distances = subset[f"{tracker}_jump_px"].dropna()
            if distances.empty:
                continue
            # A shared bin range makes the two tracker distributions comparable.
            axis.hist(distances, bins=bins, density=True, histtype="step", linewidth=1.8,
                      color=color, label=tracker.upper())
        axis.set_title(str(bodypart))
        axis.set_xlabel("Consecutive-frame distance (pixels)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25)
        axis.legend()

    for axis in axes.flat[len(bodyparts):]:
        axis.remove()
    fig.suptitle("Keypoint movement distributions: DLC vs LitPose", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    #plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc-folder", type=Path, default=DLC_FOLDER)
    parser.add_argument("--litpose-folder", type=Path, default=LITPOSE_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=DATA_FOLDER / "comparison_results")
    parser.add_argument("--jump-threshold", type=float, default=20.0,
                        help="Consecutive-frame displacement (pixels) considered a jump.")
    parser.add_argument("--frame-width", type=float,
                        help="Video width in pixels. When supplied, count frames where at least one foot "
                             "and spine 3/tail 1/tail 2 occupy opposite frame halves.")
    args = parser.parse_args()

    dlc_files = {_video_key(p): p for p in args.dlc_folder.glob("*.csv")}
    litpose_files = {_video_key(p): p for p in args.litpose_folder.glob("*.csv")}
    shared_videos = sorted(dlc_files.keys() & litpose_files.keys())
    if not shared_videos:
        raise FileNotFoundError("No matching DLC/LitPose CSV filenames were found.")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    all_results = []
    side_events = []
    for video in shared_videos:
        if args.frame_width is not None:
            for tracker, path in (("dlc", dlc_files[video]), ("litpose", litpose_files[video])):
                event_frames = count_hind_body_opposite_side_frames(read_keypoints(path), args.frame_width)
                event_frames.insert(0, "tracker", tracker)
                event_frames.insert(0, "video", path.stem)
                side_events.append(event_frames)
        result = compare_video(dlc_files[video], litpose_files[video], args.jump_threshold)
        if result.empty:
            print(f"Skipping {video}: no overlapping frames/bodyparts")
            continue
        all_results.append(result)

    if not all_results:
        raise ValueError("Matched files did not contain overlapping keypoints.")
    results = pd.concat(all_results, ignore_index=True)
    events = results.loc[results.dlc_is_jump | results.litpose_is_jump].copy()
    events.to_csv(args.output_folder / "keypoint_jump_events.csv", index=False)
    summary = (results.groupby(["video", "bodypart"], as_index=False)
               .agg(frames=("frame", "size"), dlc_jumps=("dlc_is_jump", "sum"),
                    litpose_jumps=("litpose_is_jump", "sum"),
                    jump_disagreements=("jump_disagreement", "sum"),
                    median_tracker_difference_px=("tracker_difference_px", "median")))
    summary.to_csv(args.output_folder / "keypoint_jump_summary.csv", index=False)
    if side_events:
        opposite_side_events = pd.concat(side_events, ignore_index=True)
        opposite_side_events.to_csv(args.output_folder / "hind_body_opposite_side_frames.csv", index=False)
        opposite_side_summary = (opposite_side_events.groupby(["video", "tracker", "feet_side"], as_index=False)
                                 .size().rename(columns={"size": "frame_count"}))
        opposite_side_summary.to_csv(args.output_folder / "hind_body_opposite_side_summary.csv", index=False)
        print(f"Found {len(opposite_side_events)} opposite-side hind-body frame(s).")
    save_distance_distribution(results, args.output_folder / "consecutive_frame_distance_distributions.png")
    print(f"Compared {len(shared_videos)} video(s); wrote {len(events)} jump events and a distance plot to {args.output_folder}")


if __name__ == "__main__":
    main()
