# manually inspect CAIMAN extracted neurons and save results

import h5py
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import numpy as np
import matplotlib
from skimage import exposure
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, ttk, messagebox
import glob
import os
import tifffile
import cv2

from matplotlib.patches import PathPatch
import sys
import pickle
import re
import time
#from utils_HW import *

# Load data


class ROIReviewerTk:
    def __init__(self, master, filename):
        self.master = master
        self.master.title("ROI Reviewer (No Qt)")

        # Load data
        f = load_CNMF(filename)
        pattern = r'M\d{3}_\d{6}_trial(?:[1-9]|1[0-2])(?!\d)'
        match = re.search(pattern, filename)
        self.path_name = os.path.dirname(filename)
        parent_dir = os.path.dirname(self.path_name)
        self.tiff_dir = os.path.join(parent_dir, 'motion_corrected_tiffs')
        self.tiff_files = sorted(glob.glob(os.path.join(self.tiff_dir, "*.tif")))

        # if not match:
        #     self.ses_name = None
        #     fovFilename = glob.glob(os.path.join(self.path_name,'template.tif'))
        #     self.fov = tifffile.imread(fovFilename)
        # else:
        #     self.ses_name = match.group()
        #     fovFilename = glob.glob(os.path.join(self.path_name,self.ses_name+'*template.pickle'))
        #     with open(fovFilename[0], 'rb') as f_temp:
        #         tempData = pickle.load(f_temp)
        #     self.fov = tempData['template']
        self.fov = f.estimates.Cn

        
        # image = tifffile.imread(fovFilename[0])
        #
        # image_float = image.astype(float)
        # image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min())
        #
        # # Apply histogram equalization
        # image_eq = exposure.equalize_hist(image_float)
        #
        # # Convert back to uint8 for display/saving (optional)
        # self.fov= (image_eq * 255).astype(np.uint8)
        # read the pickle file


        self.dff = f.estimates.F_dff
        self.A = f.estimates.A
        self.accepted = f.estimates.idx_components
        self.rejected = f.estimates.idx_components_bad

        self.height, self.width = self.fov.shape
        #self.height = int(self.height/2)
        #self.width = int(self.width/2)

        self.roi_type = 'Accepted'  # or 'Rejected'
        self.roi_list = self.accepted
        self.current_index = 0

        # Setup GUI widgets

        # Top frame for controls
        control_frame = tk.Frame(master)
        control_frame.pack(pady=5, fill=tk.X)

        tk.Label(control_frame, text="Show ROI Type:").pack(side=tk.LEFT, padx=(5, 2))

        self.type_var = tk.StringVar(value='Accepted')
        self.type_dropdown = ttk.Combobox(control_frame, textvariable=self.type_var,
                                          values=['Accepted', 'Rejected'], width=12, state='readonly')
        self.type_dropdown.pack(side=tk.LEFT)
        self.type_dropdown.bind("<<ComboboxSelected>>", self.on_type_change)

        tk.Label(control_frame, text="Jump to ROI #").pack(side=tk.LEFT, padx=(10, 2))
        self.jump_entry = tk.Entry(control_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind('<Return>', self.on_jump_roi)

        # Status label
        self.status = tk.Label(master, text="", font=('Arial', 12))
        self.status.pack(pady=5)

        # Figure and canvas (larger size)
        self.fig = plt.figure(figsize=(12, 9))

        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=[1.1, 1],
            width_ratios=[1.2, 1]
        )

        # Top full width
        self.ax_img = self.fig.add_subplot(gs[0, :])

        # Bottom left
        self.ax_video = self.fig.add_subplot(gs[1, 0])

        # Bottom right
        self.ax_trace = self.fig.add_subplot(gs[1, 1])

        plt.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        self.video_label = tk.Label(master)
        self.video_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Accept / Reject buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)

        self.btn_play = tk.Button(button_frame, text="Play",
                          width=20, command=self.play_active_segment)
        self.btn_play.pack(side=tk.LEFT, padx=10)

        self.btn_accept = tk.Button(button_frame, text="Accept", width=12, command=self.accept_roi)
        self.btn_accept.pack(side=tk.LEFT, padx=10)

        self.btn_quit = tk.Button(button_frame, text="Quit", width=12, command=self.master.quit)
        self.btn_quit.pack(side=tk.LEFT, padx=10)

        self.btn_reject = tk.Button(button_frame, text="Reject (space)", width=12, command=self.reject_roi)
        self.btn_reject.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(button_frame, text="Next ROI (D)", width=12, command=self.next_roi)
        self.btn_next.pack(side=tk.LEFT, padx=10)

        self.btn_prev = tk.Button(button_frame, text="Prev ROI (A)", width=12, command=self.prev_roi)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(button_frame, text="Save", width=12, command=self.save_results)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        self.btn_save_all = tk.Button(button_frame, text="Save All ROI Plots", width=18, command=self.plot_rois)
        self.btn_save_all.pack(side=tk.LEFT, padx=10)

        self.update_roi_list()
        self.update_plot()

        self.master.bind('<Key>', self.handle_keypress)

    def handle_keypress(self, event):
        key = event.keysym.lower()
        if key == 'a':
            self.prev_roi()
        elif key == 'd':
            self.next_roi()
        elif key == 'space':
            self.reject_roi()

    def update_roi_list(self):
        if self.roi_type == 'Accepted':
            self.roi_list = self.accepted
        else:
            self.roi_list = self.rejected

        if len(self.roi_list) == 0:
            self.current_index = -1
        else:
            self.current_index = 0

    def next_roi(self):
        if len(self.roi_list) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.roi_list)
        self.update_plot()

    def prev_roi(self):
        if len(self.roi_list) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.roi_list)
        self.update_plot()

    def on_type_change(self, event=None):
        self.roi_type = self.type_var.get()
        self.update_roi_list()
        self.update_plot()

    def on_jump_roi(self, event=None):
        val = self.jump_entry.get()
        if not val.isdigit():
            messagebox.showerror("Invalid Input", "Please enter a valid ROI number.")
            return
        idx = int(val) - 1
        if idx < 0 or idx >= len(self.roi_list):
            messagebox.showerror("Out of Range", f"ROI number must be between 1 and {len(self.roi_list)}.")
            return
        self.current_index = idx
        self.update_plot()

    def get_current_idx(self):
        if self.current_index == -1:
            return None
        return self.roi_list[self.current_index]

    def update_plot(self):
        self.ax_img.clear()
        self.ax_trace.clear()

        if self.current_index == -1:
            self.status.config(text=f"No ROIs in {self.roi_type} list.")
            self.btn_accept.config(state=tk.DISABLED)
            self.btn_reject.config(state=tk.DISABLED)
            self.canvas.draw()
            return

        idx = self.get_current_idx()
        self.status.config(text=f"{self.roi_type} ROI #{self.current_index + 1} / {len(self.roi_list)} (Index: {idx})")

        # Extract ROI footprint, convert sparse to dense if needed
        roi_sparse = self.A[:, idx].reshape(self.height, self.width, order='F')
        if hasattr(roi_sparse, "toarray"):
            roi = roi_sparse.toarray()
        else:
            roi = roi_sparse

        self.ax_img.imshow(self.fov, cmap='hot')

        # Fill ROI area with translucent red
        self.ax_img.imshow(roi, cmap='Reds', alpha=0.3)

        # Draw contour line on ROI boundary (level slightly above 0)
        contours = self.ax_img.contour(roi, levels=[roi[roi > 0].min()], colors='red', linewidths=3)

        self.ax_img.set_title("Field of View with ROI Contour")

        self.ax_trace.plot(self.dff[idx][0:2000], 'k')
        self.ax_trace.set_ylabel('DF/F')
        self.ax_trace.set_title("DF/F Trace")

        self.canvas.draw()

        self.btn_accept.config(state=tk.NORMAL)
        self.btn_reject.config(state=tk.NORMAL)

    def accept_roi(self):
        if self.roi_type == 'Rejected':
            idx = self.roi_list[self.current_index]
            self.accepted = np.append(self.accepted, idx)
            self.rejected = np.delete(self.rejected, np.where(self.rejected==idx))
            # Stay on the same index but now on rejected list
            if self.current_index >= len(self.roi_list):
                self.current_index = len(self.roi_list) - 1
            self.update_plot()
        else:
            # Already accepted, maybe notify user
            messagebox.showinfo("Info", "This ROI is already accepted.")

    def reject_roi(self):
        if self.roi_type == 'Accepted':
            idx = self.roi_list[self.current_index]
            self.rejected = np.append(self.rejected, idx)
            self.accepted= np.delete(self.accepted, np.where(self.accepted == idx))
            if self.current_index >= len(self.roi_list):
                self.current_index = len(self.roi_list) - 1
            self.update_plot()
        else:
            messagebox.showinfo("Info", "This ROI is already rejected.")

    def save_results(self):
        # save the updated accepted ROI in h5 format
        # variable to save: 1) df/f; 2) ROI mask
        savefilepath = os.path.join(self.path_name, self.ses_name+'_dff.h5')
        with h5py.File(savefilepath, 'w') as f:
            f.create_dataset('dff', data = self.dff[self.accepted,:])
            A_sub = self.A[:, self.accepted]
            if hasattr(A_sub, 'toarray'):
                A_sub = A_sub.toarray()

            f.create_dataset('mask', data=A_sub)

    def plot_rois(self):
        # plot all accepted rois
        save_dir = os.path.join(self.path_name, f"{self.ses_name}_ROI_figures")
        os.makedirs(save_dir, exist_ok=True)

        for i, idx in enumerate(self.accepted):
            fig, (ax_img, ax_trace) = plt.subplots(2, 1, figsize=(6, 7))
            plt.tight_layout(pad=3.0)

            # Get ROI mask
            roi_sparse = self.A[:, idx].reshape(self.height, self.width, order='F')
            roi = roi_sparse.toarray() if hasattr(roi_sparse, "toarray") else roi_sparse

            # Plot FOV + ROI contour
            ax_img.imshow(self.fov, cmap='hot')
            ax_img.imshow(roi, cmap='Reds', alpha=0.3)
            if np.any(roi > 0):
                ax_img.contour(roi, levels=[roi[roi > 0].min()], colors='red', linewidths=2)
            ax_img.set_title(f"ROI {idx} Contour")
            ax_img.axis('off')

            # Plot DF/F trace
            ax_trace.plot(self.dff[idx][0:3000], color='k', linewidth=0.8)
            ax_trace.set_title("DF/F Trace")
            ax_trace.set_xlabel("Frame")
            ax_trace.set_ylabel("ΔF/F")

            # Save figure
            fig_path = os.path.join(save_dir, f"ROI_{idx}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

        messagebox.showinfo("Done", f"Saved {len(self.accepted)} ROI plots to:\n{save_dir}")

    def get_most_active_window(self, trace, window=600):
        trace = np.array(trace)

        if len(trace) < window:
            return 0

        # moving window sum
        cumsum = np.cumsum(np.insert(trace, 0, 0))
        moving_sum = cumsum[window:] - cumsum[:-window]

        start = np.argmax(moving_sum)
        return start

    def get_roi_center_fullres(self, idx):
        roi_sparse = self.A[:, idx].reshape(self.height, self.width, order='F')
        roi = roi_sparse.toarray() if hasattr(roi_sparse, "toarray") else roi_sparse

        y, x = np.where(roi > 0)
        cy = int(np.mean(y))
        cx = int(np.mean(x))

        # upscale 300x300 → 600x600
        return cy * 2, cx * 2

    def get_roi_mask_fullres(self, idx):
        roi_sparse = self.A[:, idx].reshape(self.height, self.width, order='F')
        roi = roi_sparse.toarray() if hasattr(roi_sparse, "toarray") else roi_sparse

        # Upscale 300x300 → 600x600
        roi_full = np.kron(roi, np.ones((2, 2)))

        return roi_full


    # def play_active_segment_opencv(segment_crop, roi_mask, downsample_factor=2):
    #     """
    #     Play a small video segment with OpenCV, overlaying ROI contour.
        
    #     segment_crop: np.array [frames, height, width] (tiff crop)
    #     roi_mask: np.array [height, width] (binary)
    #     downsample_factor: spatial downsampling factor between Cn and video
    #     """

    #     # Scale ROI mask to original video size if needed
    #     if downsample_factor != 1:
    #         roi_mask_big = cv2.resize(roi_mask.astype(np.uint8),
    #                                 (roi_mask.shape[1]*downsample_factor,
    #                                 roi_mask.shape[0]*downsample_factor),
    #                                 interpolation=cv2.INTER_NEAREST)
    #     else:
    #         roi_mask_big = roi_mask

    #     # Find contours
    #     contours, _ = cv2.findContours(roi_mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Create window
    #     cv2.namedWindow('ROI Video', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow('ROI Video', 600, 600)

    #     for i in range(segment_crop.shape[0]):
    #         frame = segment_crop[i].astype(np.float32)

    #         # normalize to 0-255
    #         frame = 255 * (frame - frame.min()) / (frame.max() - frame.min())
    #         frame = frame.astype(np.uint8)

    #         # Convert to BGR for contour overlay
    #         frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #         # Draw ROI contour in red
    #         cv2.drawContours(frame_color, contours, -1, (0, 0, 255), 1)

    #         cv2.imshow('ROI Video', frame_color)

    #         key = cv2.waitKey(20)  # 20ms ~ 50 fps
    #         if key == 27:  # ESC to quit early
    #             break

    #     cv2.destroyWindow('ROI Video')

    # def play_roi_video(self):
    #     idx = self.get_current_idx()
    #     roi_mask = self.A[:, idx].reshape(self.height, self.width, order='F')
    #     segment_crop = self.load_active_segment(idx)  # your function to load 600 frames

    #     play_active_segment_opencv(segment_crop, roi_mask, downsample_factor=2)

    def play_active_segment(self):
        idx = self.get_current_idx()
        if idx is None:
            return

        trace = self.dff[idx]
        start_frame = self.get_most_active_window(trace, window=600)
        end_frame = start_frame + 600

        cy, cx = self.get_roi_center_fullres(idx)

        crop_size = 200
        half = crop_size // 2

        # ---- Determine frames per TIFF ----
        with tifffile.TiffFile(self.tiff_files[0]) as tif:
            frames_per_file = len(tif.pages)

        start_file = start_frame // frames_per_file
        end_file = (end_frame - 1) // frames_per_file

        segment_frames = []

        for file_idx in range(start_file, end_file + 1):
            tf_path = self.tiff_files[file_idx]
            with tifffile.TiffFile(tf_path) as tif:
                file_start_global = file_idx * frames_per_file
                local_start = max(start_frame - file_start_global, 0)
                local_end = min(end_frame - file_start_global, frames_per_file)

                for i in range(local_start, local_end):
                    frame = tif.asarray(key=i)
                    segment_frames.append(frame)

        segment = np.array(segment_frames)

        # ---- Crop region ----
        y1 = max(cy - half, 0)
        y2 = min(cy + half, 600)
        x1 = max(cx - half, 0)
        x2 = min(cx + half, 600)

        segment_crop = segment[:, y1:y2, x1:x2]

        # Normalize video once
        segment_crop = segment_crop.astype(np.float32)
        segment_crop -= segment_crop.min()
        segment_crop /= (segment_crop.max() + 1e-8)
        segment_crop = (segment_crop * 255).astype(np.uint8)

        # ---- Prepare ROI masks (draw ONCE) ----
        selected_mask = self.get_roi_mask_fullres(idx)
        selected_crop = selected_mask[y1:y2, x1:x2]

        nearby_masks = []
        for other_idx in self.accepted:
            if other_idx == idx:
                continue
            mask = self.get_roi_mask_fullres(other_idx)
            mask_crop = mask[y1:y2, x1:x2]
            if np.any(mask_crop > 0):
                nearby_masks.append(mask_crop)

        # ----------------------------------------------------
        # PLAY VIDEO USING OPENCV
        # ----------------------------------------------------
        #cv2.namedWindow('Active ROI Video', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Active ROI Video', crop_size, crop_size)

        for i in range(segment_crop.shape[0]):
            frame = segment_crop[i]

            # Convert to RGB
            frame_color = np.stack([frame]*3, axis=-1)

            # Draw selected ROI in red
            contours, _ = cv2.findContours(selected_crop.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_color, contours, -1, (255, 0, 0), 1)

            # Draw nearby ROIs in cyan
            for mask_crop in nearby_masks:
                contours, _ = cv2.findContours(mask_crop.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_color, contours, -1, (0, 255, 255), 1)

            # Convert to PIL Image and display in label
            img = Image.fromarray(frame_color)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # keep reference
            self.video_label.config(image=imgtk)

            # Update DF/F trace
            self.ax_trace.clear()
            self.ax_trace.plot(trace[start_frame:start_frame+600], color='k')
            self.ax_trace.axvline(i, color='red')
            self.canvas.draw_idle()
            self.master.update_idletasks()
            time.sleep(0.05)  # ~50 fps
        
        #cv2.destroyWindow('Active ROI Video')

# Run GUI
if __name__ == '__main__':
    root = tk.Tk()
    tk.Tk().withdraw()  # Hide extra prompt
    file = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
    if file:
        app = ROIReviewerTk(root, file)
        root.mainloop()


