# script using CAIMAN to batch process miniscope recordings
# including motion correction and signal extraction

#%% import

import bokeh.plotting as bpl
import cv2
import glob
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt


import glob
import matplotlib
matplotlib.use('Agg')
import time
import warnings
import pickle
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy import sparse

#%% function define
def select_rois_by_polygon(template, A, idx_components):
    from matplotlib.widgets import PolygonSelector
    from matplotlib.path import Path

    n_components = A.shape[1]
    height, width = template.shape
    A_reshaped = A.toarray().reshape((height, width, n_components))

    centroids = []
    for i in range(n_components):
        mask = A_reshaped[:, :, i]
        ys, xs = np.where(mask > mask.max() * 0.5)
        if len(xs) > 0:
            centroids.append((xs.mean(), ys.mean()))
        else:
            centroids.append((-1, -1))
    centroids = np.array(centroids)

    original_accepted = set(idx_components)
    state = {"current_accepted": set(original_accepted)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(template, cmap='gray')

    permanently_rejected = set(range(n_components)) - original_accepted
    sc_rejected = ax.scatter(
        centroids[list(permanently_rejected), 0],
        centroids[list(permanently_rejected), 1],
        c='red', s=30, label='Initially Rejected'
    )

    scatter_accepted = ax.scatter(
        centroids[list(original_accepted), 0],
        centroids[list(original_accepted), 1],
        c='green', s=30, label='Accepted'
    )

    def onselect(verts):
        path = Path(verts)
        selected = [i for i in original_accepted if path.contains_point(centroids[i])]
        state["current_accepted"] = set(selected)
        scatter_accepted.set_offsets(centroids[list(state["current_accepted"])])
        fig.canvas.draw_idle()
        fig.canvas.stop_event_loop()  # Unblock after selection

    polygon_selector = PolygonSelector(ax, onselect, useblit=True,
                                       props=dict(color='yellow', linewidth=2, alpha=0.8))
    ax.set_title("Draw polygon to select ROIs. Close or finish to proceed.")
    ax.legend()

    # Block here until polygon is drawn
    fig.canvas.start_event_loop(timeout=-1)

    plt.close(fig)

    final_accepted = sorted(state["current_accepted"])
    newly_rejected = sorted(original_accepted - state["current_accepted"])

    return final_accepted, newly_rejected

def redraw():
    global sc_accepted
    if sc_accepted is not None:
        sc_accepted.remove()
    accepted_coords = centroids[list(state["current_accepted"])]
    scatter_accepted = ax.scatter(
        accepted_coords[:, 0], accepted_coords[:, 1],
        c='green', s=30, label='Accepted'
    )
    fig.canvas.draw_idle()


#%% set up logging
logfile = None # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
# Set to logging.INFO if you want much output, potentially much more output
logger.setLevel(logging.WARNING)
logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
if logfile is not None:
    handler = logging.FileHandler(logfile)
else:
    handler = logging.StreamHandler()
handler.setFormatter(logfmt)
logger.addHandler(handler)

# set env variables in case they weren't already set
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


#%% load data
root_dir = r'/media/linda/WD_red_4TB/CalciumImage/Rotarod/batch_process/'

dataFolder = os.path.join(root_dir, 'Raw')
analysisFolder = os.path.join(root_dir, 'Analysis')
if not os.path.exists(analysisFolder):
    os.makedirs(analysisFolder)

sessions = glob.glob(os.path.join(dataFolder,'*.avi'))
nFiles = len(sessions)

memmapFile = []
sessionNames = [os.path.splitext(os.path.basename(ses))[0] for ses in sessions]
CNMFE_Model = []
templates = []

#%% set up parallel process

print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
num_processors_to_use = None

##%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'cluster' in locals():  # 'locals' contains list of current local variables
    print('Closing previous cluster')
    cm.stop_server(dview=cluster)
print("Setting up new cluster")
_, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing',
                                                 n_processes=num_processors_to_use,
                                                 ignore_preexisting=False)
print(f"Successfully set up cluster with {n_processes} processes")


#%% define parameters
frate = 30                       # movie frame rate
decay_time = 0.1                # length of a typical transient in seconds

# motion correction parameters
motion_correct = True   # flag for performing motion correction
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (5, 5)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
max_shifts = (20, 20)      # maximum allowed rigid shift
strides = (60, 60)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
max_deviation_rigid = 6  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries

# batch motion process
for idx,ses in enumerate(sessions):
    movie_path = ses
    mc_dict = {
        'fnames': movie_path,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    parameters = params.CNMFParams(params_dict=mc_dict)

    #%% motion correction
    if motion_correct:

        start = time.time()
        # do motion correction rigid
        mot_correct = MotionCorrect(movie_path, dview=cluster, **parameters.get_group('motion'))
        mot_correct.motion_correct(save_movie=True)
        fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                         np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
            # Plot shifts
            plt.figure()
            plt.plot(mot_correct.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
            savefigfile = os.path.join(analysisFolder, sessionNames[idx]+'_motion_correction.png' )
            plt.savefig(savefigfile, dpi=300)
            #plt.gcf().set_size_inches(6,3)

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
        memmapFile.append(fname_new)

        end = time.time()
        elapsed = end-start
        print(f"Time for motion correction: {elapsed:.2f} seconds")

    else:  # if no motion correction just memory map the file
        bord_px =0
        fname_new = cm.save_memmap([movie_path], base_name='memmap_',
                                   order='C', border_to_0=0, dview=cluster)

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # save motion_corrected movies
    movie_corrected = cm.load(fname_new)
    motion_corrected_path = os.path.join(analysisFolder,sessionNames[idx] + '_MC.avi');
    movie_corrected.save(motion_corrected_path)

    # %% parameters for CNMFE

    # parameters for source extraction and deconvolution
    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array([5, 5])  # expected half-width of neurons in pixels
    gSiz = 2 * gSig + 1  # half-width of bounding box created around neurons during initialization
    merge_thr = .7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 10  # amount of overlap between the patches in pixels
    tsub = 1  # downsampling factor in time for initialization, increase if you have memory problems
    ssub = 2  # downsampling factor in space for initialization, increase if you have memory problems
    gnb = 0  # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None  # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = 0  # number of background components (rank) per patch (0 for CNMFE)
    min_corr = .6  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    parameters.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                          'K': K,
                                          'gSig': gSig,
                                          'gSiz': gSiz,
                                          'merge_thr': merge_thr,
                                          'p': p,
                                          'tsub': tsub,
                                          'ssub': ssub,
                                          'rf': rf,
                                          'stride': stride_cnmf,
                                          'only_init': True,  # set it to True to run CNMF-E
                                          'nb': gnb,
                                          'nb_patch': nb_patch,
                                          'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                          'low_rank_background': low_rank_background,
                                          'update_background_components': True,
                                          # sometimes setting to False improve the results
                                          'min_corr': min_corr,
                                          'min_pnr': min_pnr,
                                          'normalize_init': False,  # just leave as is
                                          'center_psf': True,  # True for 1p
                                          'ssub_B': ssub_B,
                                          'ring_size_factor': ring_size_factor,
                                          'del_duplicates': True,  # whether to remove duplicates from initialization
                                          'border_pix': bord_px});  # number of pixels to not consider in the borders)

    cnmfe_model = cnmf.CNMF(n_processes=n_processes,
                            dview=cluster,
                            params=parameters)

    min_corr_new = 0.7
    min_pnr_new = 3

    cnmfe_model.params.change_params(params_dict={'gSig': gSig,
                                                  'min_corr': min_corr_new,
                                                  'min_pnr': min_pnr_new});

    start = time.time()

    warnings.filterwarnings("ignore")
    cnmfe_model.fit(images)
    elapsed = time.time() - start
    print(f"Time for signal extraction: {elapsed:2f} seconds")

    #%% quality checks

    min_SNR = 2  # SNR threshold
    rval_thr = 0.85  # spatial correlation threshold

    quality_params = {'min_SNR': min_SNR,
                      'rval_thr': rval_thr,
                      'use_cnn': False}
    cnmfe_model.params.change_params(params_dict=quality_params)

    cnmfe_model.estimates.evaluate_components(images, cnmfe_model.params, dview=cluster)

    print('*****')
    print(f"Total number of components: {len(cnmfe_model.estimates.C)}")
    print(f"Number accepted: {len(cnmfe_model.estimates.idx_components)}")
    print(f"Number rejected: {len(cnmfe_model.estimates.idx_components_bad)}")

    # calculate dFF
    if cnmfe_model.estimates.F_dff is None:
        print('Calculating estimates.F_dff')
        cnmfe_model.estimates.detrend_df_f(quantileMin=8,
                                           frames_window=250,
                                           flag_auto=False,
                                           use_residuals=False);  # use denoised data
    else:
        print("estimates.F_dff already defined")

    # calculate correlation image

    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(images[::max(T // 200, 1)],
                                                                               # subsample if needed
                                                                               gSig=gSig[0],  # used for filter
                                                                               swap_dim=False)

    #%% save template
    # b: baseline background
    # mean intensity - background
    template = np.nanmean(images, 0) - np.reshape(cnmfe_model.estimates.b0, (600, 600), order='F')
    templates.append(template)

    # save in pickle # also save the plot
    savefilename = os.path.join(analysisFolder, sessionNames[idx] + '_reg_template.pickle')
    data = {'template': template}
    with open(savefilename, 'wb') as f:
        pickle.dump(data, f)

    saveplotname = os.path.join(analysisFolder, sessionNames[idx]  + '_reg_template.png')
    saveplottiff = os.path.join(analysisFolder, sessionNames[idx]  + '_reg_template.tif')
    plt.figure()
    plt.imshow(template)
    plt.savefig(saveplotname)
    plt.savefig(saveplottiff)

    # %% plot contour plots of accepted and rejected components
    cnmfe_model.estimates.plot_contours(img=template,
                                           idx=cnmfe_model.estimates.idx_components, display_numbers=False)

    fig = plt.gcf()  # Get current figure
    fig.savefig(os.path.join(analysisFolder, sessionNames[idx]  + '_contours.png'), dpi=300)

    save_results = True
    if save_results:
        # save_path = os.path.join(root_dir, session+'_results.hdf5')
        save_path = os.path.join(analysisFolder, sessionNames[idx]+'_results.hdf5')
        # save_path =  r'demo_pipeline_cnmfe_results.hdf5'  # or add full/path/to/file.hdf5
        cnmfe_model.estimates.Cn = correlation_image  # squirrel away correlation image with cnmf object
        cnmfe_model.save(save_path)

    CNMFE_Model.append(cnmfe_model)

#%% go over every session, manually exclude the surrounding false positive ROIs
#import matplotlib
matplotlib.use('TkAgg')

for ii in range(nFiles):
    template = templates[ii]
    cnmfe_model = CNMFE_Model[ii]
    A = cnmfe_model.estimates.A
    idx_components = set(cnmfe_model.estimates.idx_components)

    # 🔻 BLOCKS here until user finishes drawing polygon
    final_accepted, newly_rejected = select_rois_by_polygon(template, A, idx_components)

    print("Final accepted ROI indices:", final_accepted)
    print("Newly rejected ROI indices:", newly_rejected)

    # 🔽 Update the model
    cnmfe_model.estimates.idx_components = np.array(final_accepted)
    cnmfe_model.estimates.idx_components_bad = np.concatenate([
        cnmfe_model.estimates.idx_components_bad,
        np.array(newly_rejected)
    ])

    # 🔽 Plot updated contours
    cnmfe_model.estimates.plot_contours(img=template,
        idx=cnmfe_model.estimates.idx_components, display_numbers=False)

    fig = plt.gcf()
    fig.savefig(os.path.join(analysisFolder, sessionNames[ii] + '_adjusted_contours.png'), dpi=300)

    save_path = os.path.join(analysisFolder, sessionNames[ii] + 'adjusted_results.hdf5')
    cnmfe_model.estimates.Cn = correlation_image  # squirrel away correlation image
    cnmfe_model.save(save_path)

    plt.close('all')