# batch process for caiman motion correction 
import multiprocessing as mp
import cv2
import glob
import holoviews as hv
import logging
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt

import numpy as np
import os
import psutil
import re
import gc
import time

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt

from utils_HW import *
from tqdm import tqdm
import tifffile

import matplotlib.patches as patches

try:
    cv2.setNumThreads(0)
except:
    pass



def frame_row_corr_worker(args):
    ff, mmap_file, shape, dtype, ref = args
    images = np.memmap(mmap_file, dtype=dtype, mode='r', shape=shape)
    return frame_row_corr(images[ff], ref)

def main(root_path, if_batch=False, task=None):

    # find all sessions
    # find all image videos
    if if_batch: # if batch processing, get all folders under root_path
        session_list = glob.glob(os.path.join(root_path, '**'))
    else: # single folder processing
        session_list = [root_path]

    sessions_todo = []
    ImgVideos_list = []
    ImgName_list = []
    
    if task == 'odor':
        for ses in session_list:
            # check if img video exists, and processed video does not exist
            Imgvideo = glob.glob(os.path.join(ses, '*_ImgVideo*.avi'))
            if len(Imgvideo)>0:
                ses_video_list = []   # sublist for this session
                ses_name_list = []  
                sesName = os.path.basename(Imgvideo[0])[:-4] 
                if len(Imgvideo)==1:                
                    processed_file = glob.glob(os.path.join(ses, 'temp', sesName+'*_C_frames_*.mmap'))
                else:
                    processed_file = glob.glob(os.path.join(ses, 'temp', sesName[0]+'_combined_*_C_frames_*.mmap'))  # if multiple videos, just process all
                # if more than 1 videos, concatenate them first
                if len(processed_file)==0:
                    for ii in range(len(Imgvideo)):
                        # get the file name
                        sesName = os.path.basename(Imgvideo[ii])[:-4] 
                        ses_video_list.append(Imgvideo[ii])
                        ses_name_list.append(sesName)
                        processed_file = glob.glob(os.path.join(ses, 'temp', ses+'*_C_frames_*.mmap'))
                    ImgName_list.append(ses_name_list)
                    sessions_todo.append(ses)
                    ImgVideos_list.append(ses_video_list)

    elif task == 'rotarod':
        ses_folder = []
        for ses in session_list:
            Imgvideo = glob.glob(os.path.join(ses, '*_ImgVideo*.avi'))
            if len(Imgvideo)>0:
                for vv in range(len(Imgvideo)):                    
                    sesName = os.path.basename(Imgvideo[vv])[:-4] 
                    processed_file = glob.glob(os.path.join(ses, 'temp', sesName+'*_C_frames_*.mmap'))
                # if more than 1 videos, concatenate them first

                        # get the file name
                    sesName = os.path.basename(Imgvideo[vv])[:-4] 
                    ses_folder.append(ses)
                    sessions_todo.append(sesName)
                    ImgVideos_list.append(Imgvideo[vv])

    
    #%% parallel computing setup
    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
    num_processors_to_use = None

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=num_processors_to_use, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")

    #%% go through each session
    for idx,ses in enumerate(sessions_todo):
        
        #%% ---------------- LOGGING SETUP ----------------
        #log_dir = r'/global/scratch/users/hongliwang/miniscope/test'
        if task == 'odor':
            out_path = os.path.join(ses,'temp')
            sesName = ImgName_list[idx][0][0:-9]
        elif task == 'rotarod':
            out_path = os.path.join(ses_folder[idx],'temp')
            sesName = sessions_todo[idx][0:-9]
 
        log_dir = os.path.join(ses_folder[idx],'temp')
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, 'preprocess.log')

        logger = logging.getLogger('preprocess')
        logger.setLevel(logging.INFO)

        logfmt = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(process)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        fh = logging.FileHandler(logfile)
        fh.setFormatter(logfmt)
        logger.addHandler(fh)
        logger.info("===== preprocess started =====")
        t_total_start = time.perf_counter()

        # limit threading
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


        movie_path = ImgVideos_list[idx]
        os.makedirs(out_path, exist_ok=True)
        print('Processing session: {}'.format(ses))
        
        # define parameters
        os.environ['CAIMAN_DATA']=ses_folder[idx]
        # dataset dependent parameters
        frate = 20                      # movie frame rate
        decay_time = 0.3                 # length of a typical transient in seconds

        # motion correction parameters
        motion_correct = True    # flag for performing motion correction
        pw_rigid = False        # flag for performing piecewise-rigid motion correction (otherwise just rigid)
        gSig_filt = (10, 10)       # sigma for high pass spatial filter applied before motion correction, used in 1p data1
        max_shifts = (20, 20)      # maximum allowed rigid shift
        strides = (16, 16)       # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (8, 8)      # overlap between patches (size of patch = strides + overlaps)
        max_deviation_rigid = 10  # maximum deviation allowed for patch with respect to rigid shifts
        border_nan = 'copy'      # replicate values along the boundaries

        # test gSig_filt
        # from caiman.motion_correction import high_pass_filter_space

        # filtered = high_pass_filter_space(frame, gSig_filt)
        # plt.subplot(1,2,1); plt.imshow(frame); plt.title('Raw')
        # plt.subplot(1,2,2); plt.imshow(filtered); plt.title('Filtered')

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
 
        t0 = time.perf_counter() # motion correction start time
        logger.info(f"Motion correction parameters: {mc_dict}")

        n_Iters = 1
        if motion_correct and not os.path.exists(os.path.join(out_path, sesName+'_rigid_shifts_round{}.png'.format(n_Iters))):
            # do motion correction rigid
            current_movie = movie_path
            total_bord_px = 0
            for iter in range(n_Iters):
                print(f"Motion correction iteration {iter+1}/{n_Iters}...")
                mot_correct = MotionCorrect(current_movie,
                    dview=cluster,
                    **parameters.get_group('motion'))
                mot_correct.motion_correct(save_movie=True)
                fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
                if pw_rigid:
                    bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                                np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
                else:
                    bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
                    # Plot shifts
                    plt.plot(mot_correct.shifts_rig)  # % plot rigid shifts
                    plt.legend(['x shifts', 'y shifts'])
                    plt.xlabel('frames')
                    plt.ylabel('pixels')
                    plt.gcf().set_size_inches(6,3)
                    plt.savefig(os.path.join(out_path, sesName + '_rigid_shifts_round{}.png'.format(iter+1)))
                    plt.close()

                #total_bord_px += bord_px
                current_movie = fname_mc  # use the output of current iteration as input for next iteration
                del mot_correct
                import gc
                gc.collect()
                #fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                #                           border_to_0=bord_px)

            # try pw-rigid correction
            # mot_correct.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
            # #mot_correct.template = mot_correct.mmap_file  # use the template obtained before to save in computation (optional)

            # mot_correct.motion_correct(save_movie=True, template=mot_correct.total_template_rig)
            # m_els = cm.load(mot_correct.fname_tot_els)

            # plt.figure(figsize = (20,10))
            # plt.subplot(2, 1, 1)
            # plt.plot(mot_correct.x_shifts_els)
            # plt.ylabel('x shifts (pixels)')
            # plt.subplot(2, 1, 2)
            # plt.plot(mot_correct.y_shifts_els)
            # plt.ylabel('y_shifts (pixels)')
            # plt.xlabel('frames')
            # #%% compute borders to exclude
            # bord_px_els = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
            #                                 np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
            # plt.savefig(os.path.join(out_path, sesName + '_pw_rigid_shifts2.png'))
            # plt.close()
 
        else:  # if no motion correction just memory map the file
            fname_mc = glob.glob(os.path.join(out_path, sesName+'*_rig_*_order_F*.mmap'))
        
        #%% quality control

        # final_size = np.subtract(mot_correct.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
        # winsize = 100
        # swap_dim = False
        # resize_fact_flow = .2    # downsample for computing ROF

        # tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
        #     movie_path, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        # tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
        #     mot_correct.fname_tot_rig[0], final_size[0], final_size[1],
        #     swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        # tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
        #     mot_correct.fname_tot_els[0], final_size[0], final_size[1],
        #     swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
        
        # plt.figure(figsize = (20,10))
        # plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_rig); plt.plot(correlations_els)
        # plt.legend(['Original','Rigid','PW-Rigid'])
        # plt.subplot(223); plt.scatter(correlations_orig, correlations_rig); plt.xlabel('Original'); 
        # plt.ylabel('Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
        # axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
        # plt.subplot(224); plt.scatter(correlations_rig, correlations_els); plt.xlabel('Rigid'); 
        # plt.ylabel('PW-Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
        # axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
        # plt.savefig(os.path.join(out_path, sesName + '_mc_comparison.png'))


        



        #%% 
        print('Finished processing session: {}'.format(ses))
        logger.info(f"Motion correcttion finished in ({time.perf_counter() - t0:.2f} s)")

        #%% after motion correction, find the memmap file in F order and save it in C order and tiffs

        t0 = time.perf_counter() # time for saving tiffs and calculate row correlation
        logger.info(f"Saving motion corrected tiffs and calculating row correlation started.")

        print('Saving motion corrected video in tiffs...')
        output_tiff_dir = os.path.join(ses_folder[idx], 'motion_corrected_tiffs') 
        tiff_files = glob.glob(os.path.join(output_tiff_dir, '*.tif')) + \
             glob.glob(os.path.join(output_tiff_dir, '*.tiff'))

        # write tiff files, also compute row_correlation to check for screen tearing simultaneously
        #box_size = 500   # final output size  # crop size around selected center
        #crop_size = 600
        if not len(tiff_files) > 20: # some arbitrary number
        # load memmap (NO RAM copy)
            prev_nTiff = 0  # keep track of previous number of tiffs
            for mIdx, mmap_file in enumerate(fname_mc):


                Yr, dims, T = cm.load_memmap(mmap_file)
                images = Yr.T.reshape((T,) + dims, order='F')
                shape = (T,) + dims
                dtype = Yr.dtype
                mmap_file = fname_mc[mIdx]
                nFrames = images.shape[0]
                d1, d2 = dims[:2]

                if mIdx == 0 and idx == 0:
                    first_frame = Yr[:, 0].reshape(d1, d2, order='F')

                # Inline ROI selection variables
                #     roi_coords = []

                #     fig, ax = plt.subplots()
                #     ax.imshow(first_frame, cmap='gray')
                #     ax.set_title("Drag the red box to set ROI, then close the window to confirm")

                #     # Starting rectangle position (top-left corner)
                #     x0, y0 = 100, 100  # initial position
                #     rect = patches.Rectangle(
                #         (x0, y0), box_size, box_size,
                #         linewidth=2, edgecolor='r', facecolor='none'
                #     )
                #     ax.add_patch(rect)
                #     dr = DraggableRectangle(rect)

                #     # Show the plot; user drags rectangle, then closes figure to confirm
                #     plt.show()

                #     # After closing the figure, get final coordinates
                #     x_final, y_final = rect.get_xy()
                #     x1, y1 = int(x_final), int(y_final)
                #     x2, y2 = x1 + box_size, y1 + box_size

                # cx = (x1 + x2) // 2
                # cy = (y1 + y2) // 2
                # x_start = max(cx - crop_size // 2, 0)
                # y_start = max(cy - crop_size // 2, 0)
                # x_end = min(x_start + crop_size, d2)
                # y_end = min(y_start + crop_size, d1)


                # CHANGE THIS
                frames_per_tiff = 1000
                n_tiffs = nFrames//frames_per_tiff+1
                dtype_out = np.uint16

                os.makedirs(output_tiff_dir, exist_ok=True)

                # define reference frame for row correlation
            # Load memmap (disk-backed)
            

                for i in tqdm(range(n_tiffs)):
                    start = int(i * frames_per_tiff)
                    end = int(min((i + 1) * frames_per_tiff, T))

                    if start >= T:
                        break

                    out_path_tt = os.path.join(
                        output_tiff_dir,
                        sesName + f'_motion_corrected_part_{(i+prev_nTiff):03d}.tif'
                    )
                    

                    with tifffile.TiffWriter(out_path_tt, bigtiff=True) as tif:
                        # iterate in smaller chunks inside each TIFF
                        for s in range(start, end, 1000):
                            e = min(s + 1000, end)

                            # memmap slice (NO full RAM copy)
                            Y_chunk = Yr[:, s:e].reshape((d1, d2, e - s), order='F').transpose(2,0,1)
                            #Y_chunk_crop = Y_chunk[:, y1:y2, x1:x2]

  
                            #Y_chunk_resized = np.empty((Y_chunk_crop.shape[0], box_size, box_size), dtype=Y_chunk_crop.dtype)
                            # for f in range(Y_chunk_crop.shape[0]):
                            #     Y_chunk_resized[f] = cv2.resize(Y_chunk_crop[f], (box_size, box_size), interpolation=cv2.INTER_AREA)

                            # Save to TIFF
                            # tif.write(
                            #     Y_chunk_crop.astype(dtype_out, copy=False),
                            #     photometric='minisblack'
                            # )
                            tif.write(
                                Y_chunk.astype(dtype_out, copy=False),
                                photometric='minisblack'
                            )
                # fig, ax = plt.subplots()
                # ax.imshow(first_frame, cmap='gray')
                # ax.set_title(f"{sesName} - Part {(i+prev_nTiff):03d}")

                # 600x600 crop rectangle (blue)

                # 400x400 ROI rectangle (red)
                # roi_rect = patches.Rectangle(
                #     (x1, y1), box_size, box_size,
                #     linewidth=2, edgecolor='r', facecolor='none', label='400x400 ROI'
                # )
                # ax.add_patch(roi_rect)

                # ax.legend()
                # png_path = os.path.join(output_tiff_dir, sesName + f'_ROI_crop_part_{(i+prev_nTiff):03d}.png')
                # plt.savefig(png_path, dpi=150)
                # plt.close()

                prev_nTiff = prev_nTiff + n_tiffs

        
        logger.info(f"Saving motion corrected tiffs and calculating row correlation finished in ({time.perf_counter() - t0:.2f} s)")

        #%% save the first tiff file in c-order memmap 
        # save the tiff file to C order memmap file
        t0 = time.perf_counter() # time for saving c-order memmap

        tiff_files = [
            os.path.join(output_tiff_dir, f)
            for f in os.listdir(output_tiff_dir)
            if (f.lower().endswith((".tif", ".tiff")) and sesName in f)
        ]
        #print(tiff_files)

        # save downsampled memmap only for now

        # save the bord_px value in file name


        #fname_new = cm.save_memmap(tiff_files,base_name=sesName+'_'+str(bord_px)+'px_border_', order='C',
        #                            border_to_0=bord_px)

        # spatial downsample
        bord_px = 0
        fname_DS = cm.save_memmap(tiff_files,base_name=sesName+'_ds2_'+str(bord_px)+'px_border_', order='C', resize_fact=(0.5,0.5,1),
                            border_to_0=bord_px)
        #print(tiff_files)

        
        # delete F_order memmap file and all individual c-order memmap file except for the first one
        pattern = re.compile(
            rf"^{re.escape(sesName)}.*_(\d{{4}})_d1"
        )

        for fname in os.listdir(out_path):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                if idx != 0:  # keep sesName_0000_d1*
                    fullpath = os.path.join(out_path, fname)
                    print(f"Deleting {fullpath}")
                    os.remove(fullpath)

        logger.info(f"Saved C-order memmap in ({time.perf_counter() - t0:.2f} s)")


        # clear the motion correction object
        #del mot_correct
            #plt.figure()
            # plt.imshow(images[27362], cmap='gray'
            #            )
            # plt.show()
            # plt.savefig(os.path.join(out_path, sesName+'_frame_27362.png'))

        # clean up


        # ---- release F-order memmap cleanly ----
        # del images
        # del Yr
        # del fname_mc


        # gc.collect()

        # f_order_file = glob.glob(os.path.join(out_path, '*_order_F_*.mmap'))
        # if len(f_order_file)>0:
        #     os.remove(f_order_file[0])
        #     print('Deleted F order memmap file: {}'.format(f_order_file[0]))

# ---------------- REQUIRED ON WINDOWS ----------------

class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.cid_press = rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.rect.axes: return
        contains, _ = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = (x0, y0, event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_xy((x0 + dx, y0 + dy))
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.rect.figure.canvas.draw()

if __name__ == "__main__":
    mp.freeze_support()

    # process single folder
    # root_path = r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260118'
    # if_batch = False
    # task = 'odor'
    # main(root_path, if_batch, task)

    task = 'rotarod'
    root_path = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001\Rotarod\Imaging'
    if_batch = True
    main(root_path, if_batch, task)
    # batch process
    #root_path = r'Y:\HongliWang\Miniscope\ASDC001'
    #if_batch = True
    #main(root_path, if_batch)


    ## spatially downsample the video 
    # orig_file = r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\temp\ASDC001_AB_ImgVideo_2026-01-13T09_07_50_rig__d1_600_d2_600_d3_1_order_C_frames_154469.mmap'

    # ds_factor = 2
    # Yr, dims, T = cm.load_memmap(orig_file)
    # d1, d2 = dims
    # images = Yr.T.reshape((T,) + dims, order='F')
    # os.environ['CAIMAN_DATA']=r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113'
    # # ------------------------------------------------------------------
    # # Save new mmap
    # # ------------------------------------------------------------------
    # base, _ = os.path.splitext(orig_file)
    # ds_file = 'ASDC001_AB_ImgVideo_2026-01-13T09_07_50_rig_'
    # tiff_files=  glob.glob(r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\motion_corrected_tiffs\*.tif')

    
    # dsfile = cm.save_memmap(
    #     tiff_files,
    #     base_name=ds_file,
    #     resize_fact=(0.5,0.5,1),
    #     order='C'
    # )

    # Yr, dims, T = cm.load_memmap(dsfile)
    # d1, d2 = dims
    # images = Yr.T.reshape((T,) + dims, order='F')

    # print('Saved spatially downsampled mmap:')
    # print(ds_file)
    # print(f'New shape: {new_d1} x {new_d2}, frames={T}')