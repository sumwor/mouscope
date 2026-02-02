import multiprocessing as mp
import time
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use("Qt5Agg")

import argparse
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr


def main():
    # ---------------- LOGGING SETUP ----------------
    #log_dir = r'/global/scratch/users/hongliwang/miniscope/test'
    log_dir = r'Y:\HongliWang\Miniscope\ASDC001_260113\caiman_result'
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, 'cnmfe_full.log')

    logger = logging.getLogger('caiman_run')
    logger.setLevel(logging.INFO)

    logfmt = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(process)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(logfile)
    fh.setFormatter(logfmt)
    logger.addHandler(fh)

    # also show on console if you want
    sh = logging.StreamHandler()
    sh.setFormatter(logfmt)
    logger.addHandler(sh)

    logger.info("===== CNMF-E run started =====")
    t_total_start = time.perf_counter()

        # limit threading
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


    # ---------------- CLUSTER SETUP ----------------
    t0 = time.perf_counter()
    logger.info(f"You have {psutil.cpu_count()} CPUs available")

    print(f"You have {psutil.cpu_count()} CPUs available")
    num_processors_to_use = None

    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing",
        n_processes=num_processors_to_use,
        ignore_preexisting=False
    )
    print(f"Successfully set up cluster with {n_processes} processes")

    logger.info(
        f"Cluster setup with {n_processes} processes "
        f"({time.perf_counter() - t0:.2f} s)"
        )


    # ---------------- MOTION CORRECTION PARAMS ----------------
    #movie_path = r'Z:\HongliWang\Miniscope\ASDC001_260113\ASDC001_AB_ImgVideo_2026-01-13T09_07_50.avi'
    movie_path = r'Y:\HongliWang\Miniscope\ASDC001_260113\temp\memmap_0000_d1_600_d2_600_d3_1_order_C_frames_773.mmap'
    #movie_path = r'/global/scratch/users/hongliwang/miniscope/test/memmap_0000_d1_600_d2_600_d3_1_order_C_frames_773.mmap'
    frate = 20
    decay_time = 0.3

    mc_dict = {
        'fnames': movie_path,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': False,
        'max_shifts': (20, 20),
        'gSig_filt': (5, 5),
        'strides': (160, 160),
        'overlaps': (40, 40),
        'max_deviation_rigid': 3,
        'border_nan': 'copy'
    }

    parameters = params.CNMFParams(params_dict=mc_dict)

    # ---------------- LOAD MEMMAP ----------------
    t0 = time.perf_counter()
    #fname_new = r'/global/scratch/users/hongliwang/miniscope/test/memmap__d1_600_d2_600_d3_1_order_C_frames_154469.mmap'
    fname_new = r'Y:\HongliWang\Miniscope\ASDC001_260113\temp\memmap__d1_600_d2_600_d3_1_order_C_frames_154469.mmap'

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # ---------------- CNMF-E PARAMS ----------------
    gSig = np.array([6, 6])
    gSiz = 2 * gSig + 1

    parameters.change_params(params_dict={
        'method_init': 'corr_pnr',
        'K': None,
        'gSig': gSig,
        'gSiz': gSiz,
        'merge_thr': .7,
        'p': 1,
        'tsub': 1,
        'ssub': 1,
        'rf': 40,
        'stride': 20,
        'only_init': True,
        'nb': 0,
        'nb_patch': 0,
        'method_deconvolution': 'oasis',
        'low_rank_background': None,
        'update_background_components': True,
        'min_corr': .8,
        'min_pnr': 10,
        'normalize_init': False,
        'center_psf': True,
        'ssub_B': 1,
        'ring_size_factor': 1.4,
        'del_duplicates': True,
        'border_pix': 0
    })

    # correlation image
    fname_corr = r'Y:\HongliWang\Miniscope\ASDC001_260113\temp\memmap_0000_d1_600_d2_600_d3_1_order_C_frames_773.mmap'

    Yr_corr, dims_corr, T_corr = cm.load_memmap(fname_corr)
    images_corr = Yr_corr.T.reshape((T_corr,) + dims_corr, order='F')
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(images_corr, # subsample if needed
                                                                           gSig=gSig[0], # used for filter
                                                                           swap_dim=False)
    logger.info(f"Computed correlation/PNR ({time.perf_counter() - t0:.2f} s)")

    
    # ---------------- RUN CNMF-E ----------------
    t0 = time.perf_counter()
    cnmfe_model = cnmf.CNMF(
        n_processes=n_processes,
        dview=cluster,
        params=parameters
    )

    #cnmfe_model.fit_file(fname_new)
    cnmfe_model.fit(images)
    logger.info(f"CNMF-E fit completed ({time.perf_counter() - t0:.2f} s)")

    # ROI evaluation
    t0 = time.perf_counter()
    cnmfe_model.params.change_params(params_dict={
        'min_SNR': 2,
        'rval_thr': 0.85,
        'use_cnn': False
    })

    cnmfe_model.estimates.evaluate_components(
        images, cnmfe_model.params, dview=cluster
    )
    logger.info(
        f"Component evaluation completed "
        f"({time.perf_counter() - t0:.2f} s)"
        )

    print('*****')
    print(f"Total components: {len(cnmfe_model.estimates.C)}")
    print(f"Accepted: {len(cnmfe_model.estimates.idx_components)}")
    print(f"Rejected: {len(cnmfe_model.estimates.idx_components_bad)}")
    logger.info("----- RESULTS -----")
    logger.info(f"Total components: {len(cnmfe_model.estimates.C)}")
    logger.info(f"Accepted: {len(cnmfe_model.estimates.idx_components)}")
    logger.info(f"Rejected: {len(cnmfe_model.estimates.idx_components_bad)}")

    # cnmfe_model.estimates.view_components(img=correlation_image, 
    #                                     idx=cnmfe_model.estimates.idx_components,
    #                                     cmap='viridis', #gray
    #                                     thr=.9); #increase to see full footprint


    save_results = True

    if save_results:
        #save_path =  r'/global/scratch/users/hongliwang/miniscope/test/test_small.hdf5'  # or add full/path/to/file.hdf5
        save_path = r'Y:\HongliWang\Miniscope\ASDC001_260113\caiman_result\test_full.hdf5'  # or add full/path/to/file.hdf5
        cnmfe_model.estimates.Cn = correlation_image # squirrel away correlation image with cnmf object
        cnmfe_model.save(save_path)

    logger.info(
        f"Saved results to {save_path} "
        f"({time.perf_counter() - t0:.2f} s)"
    )

    logger.info(
        f"===== CNMF-E finished | TOTAL TIME: "
        f"{time.perf_counter() - t_total_start:.2f} s ====="
    )

# ---------------- REQUIRED ON WINDOWS ----------------
if __name__ == "__main__":
    mp.freeze_support()
    main()
