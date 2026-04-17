"""
longitudinal_registration.py

Purpose
-------
Perform cross-session longitudinal registration for one animal by:
1. Collecting CNMF result files and corresponding template mmap files
   from both Rotarod and Odor imaging sessions.
2. Loading spatial footprints (A matrices) from each CNMF result file.
3. Loading mean template images from each mmap file.
4. Running CaImAn's register_multisession across sessions.
5. Saving intermediate outputs and returning registration results.

Important notes
---------------
- Now, this script is not designed for interactive GUI display.
  Therefore it uses the non-interactive matplotlib backend "Agg".
- The current code assumes each spatial footprint can be reshaped into
  (n_neurons, 300, 300), matching our current dataset.

"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib

# Use a non-interactive backend so this script can run in batch mode
# without requiring a Qt windowing environment.
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.sparse import csc_matrix

import caiman as cm
from caiman.base.rois import register_multisession
from caiman.utils import visualization


def run_longitudinal_registration(
    root_dir=r"\\filenest.dyn.berkeley.edu\Wilbrecht_file_server\HongliWang\Miniscope\ASD\Data\ASDC001",
    n_reg=7,
    max_sessions=10,
):
    """
    Run multisession registration across rotarod + odor imaging sessions.

    Parameters
    ----------
    root_dir : str
        Animal-level data directory, e.g.
        \\filenest...\ASD\Data\ASDC001
    n_reg : int
        Minimal number of sessions a component must appear in to be kept.
    max_sessions : int or None
        Only keep the first N sessions after collecting file paths.
        Use None to keep all sessions.

    Returns
    -------
    dict
        Dictionary containing registration outputs and loaded data.
    """
    print("entered run_longitudinal_registration")
    print("root_dir:", root_dir)

    cnmfe_results = []
    template_files = []

    rotarod_folder = os.path.join(root_dir, "Rotarod", "Imaging")
    odor_folder = os.path.join(root_dir, "Odor", "Imaging")

    print("rotarod exists:", os.path.exists(rotarod_folder))
    print("odor exists:", os.path.exists(odor_folder))

    if not os.path.exists(rotarod_folder):
        raise FileNotFoundError(f"Rotarod imaging folder not found: {rotarod_folder}")
    if not os.path.exists(odor_folder):
        raise FileNotFoundError(f"Odor imaging folder not found: {odor_folder}")

    date_folders_rotarod = sorted(
        [f for f in os.listdir(rotarod_folder) if os.path.isdir(os.path.join(rotarod_folder, f))]
    )
    date_folders_odor = sorted(
        [f for f in os.listdir(odor_folder) if os.path.isdir(os.path.join(odor_folder, f))]
    )

    # Collect rotarod sessions files
    for date in date_folders_rotarod:
        caiman_folder = os.path.join(rotarod_folder, date, "caiman_results")
        template_folder = os.path.join(rotarod_folder, date, "temp")

        cnmfe_file = []
        template_file = []

        if os.path.exists(caiman_folder):
            cnmfe_file = glob.glob(os.path.join(caiman_folder, "*_updated_cnmf.mat"))
        if os.path.exists(template_folder):
            template_file = glob.glob(os.path.join(template_folder, "*_0000_d1_300_d2_300_d3_1*.mmap"))

        cnmfe_results.extend(cnmfe_file)
        template_files.extend(template_file)

    # Collect odor sessions files
    for date in date_folders_odor:
        caiman_folder = os.path.join(odor_folder, date, "caiman_results")
        template_folder = os.path.join(odor_folder, date, "temp")

        cnmfe_file = []
        template_file = []

        if os.path.exists(caiman_folder):
            cnmfe_file = glob.glob(os.path.join(caiman_folder, "*updated_cnmf.mat"))
        if os.path.exists(template_folder):
            template_file = glob.glob(os.path.join(template_folder, "*_0000_d1_300_d2_300_d3_1*.mmap"))

        cnmfe_results.extend(cnmfe_file)
        template_files.extend(template_file)

    print("n cnmfe files before cutoff:", len(cnmfe_results))
    print("n template files before cutoff:", len(template_files))

# Apply cutoff early so downstream loading only touches the first N sessions.
    if max_sessions is not None:
        cnmfe_results = cnmfe_results[:max_sessions]
        template_files = template_files[:max_sessions]

    if len(cnmfe_results) != len(template_files):
        raise ValueError(
            f"Number of CNMF files ({len(cnmfe_results)}) != "
            f"number of template files ({len(template_files)})"
        )

    nFiles = len(cnmfe_results)
    print("nFiles after cutoff:", nFiles)

    if nFiles == 0:
        raise ValueError("No session files found for longitudinal registration.")

    spatial = []
    templates = []
    spatial_reshaped = []

    spatial_footprint_dir = os.path.join(root_dir, "spatial_footprint")
    os.makedirs(spatial_footprint_dir, exist_ok=True)

    for i in range(nFiles):
        print(f"processing session {i + 1}/{nFiles}")
        print("cnmfe:", cnmfe_results[i])
        print("template:", template_files[i])

        # Detect MAT version
        with open(cnmfe_results[i], "rb") as f:
            header = f.read(128)

        if b"MATLAB 7.3" in header:
            with h5py.File(cnmfe_results[i], "r") as f:
                A_grp = f["results"]["A"]
                A = csc_matrix(
                    (
                        A_grp["data"][()],
                        A_grp["ir"][()],
                        A_grp["jc"][()],
                    ),
                    shape=(90000, len(A_grp["jc"]) - 1),
                )
        else:
            cnmfe_data = loadmat(cnmfe_results[i])
            A = cnmfe_data["results"]["A"][0][0]

        spatial.append(A)

        n_neurons = A.shape[1]
        A_dense = A.toarray()
        A_reshaped = A_dense.T.reshape((n_neurons, 300, 300))
        spatial_reshaped.append(A_reshaped)

        savefigpath = os.path.join(
            spatial_footprint_dir,
            f"session{i}_spatial_footprint.mat",
        )
        sio.savemat(savefigpath, {"footprints": A_reshaped})

        Yr, dims, T = cm.load_memmap(template_files[i])
        images = Yr.T.reshape((T,) + dims, order="F")
        template = np.mean(images, axis=0)
        templates.append(template)

    # Save reshaped spatial footprints
    mat_dict = {}
    for i, arr in enumerate(spatial_reshaped):
        mat_dict[f"session_{i + 1}"] = arr

    sio.savemat(os.path.join(root_dir, "spatial_reshaped.mat"), mat_dict)

    #Run multisession registration
    dims = templates[0].shape

    print("starting register_multisession")
    spatial_union, assignments, matchings = register_multisession(
        A=spatial,
        dims=dims,
        templates=templates,
        max_thr=0.8,
        thresh_cost=1,
    )
    print("finished register_multisession")

    # Keep only components that appear in at least n_reg sessions
    assignments_filtered = np.array(
        np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]),
        dtype=int,
    )

    if assignments_filtered.size == 0:
        print("No components passed n_reg filtering.")
        spatial_filtered = None
    else:
        spatial_filtered = spatial[0][:, assignments_filtered[:, 0]]
        visualization.plot_contours(spatial_filtered, templates[-1])

        contour_fig_path = os.path.join(root_dir, "longitudinal_contours.png")
        plt.savefig(contour_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("saved contour figure:", contour_fig_path)

    return {
        "spatial_union": spatial_union,
        "assignments": assignments,
        "matchings": matchings,
        "assignments_filtered": assignments_filtered,
        "spatial_filtered": spatial_filtered,
        "templates": templates,
        "spatial": spatial,
        "spatial_reshaped": spatial_reshaped,
        "cnmfe_results": cnmfe_results,
        "template_files": template_files,
    }


if __name__ == "__main__":
    result = run_longitudinal_registration()
    print("run_longitudinal_registration finished")
    print(result.keys())