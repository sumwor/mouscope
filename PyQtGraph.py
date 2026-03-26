import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QMessageBox,
    QTextEdit, QProgressBar
)
from PyQt6.QtCore import (
    Qt, QObject, QThread, pyqtSignal
)
import numpy as np
import h5py
from PyQt6.QtWidgets import QFileDialog
import glob
import os
import re
from PyQt6.QtGui import QImage, QPixmap
import cv2
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import tifffile
import matplotlib
matplotlib.use("QtAgg") 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
plt.ion()

class LoadWorker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        import h5py, numpy as np, os, glob, re

        data = {}

        with h5py.File(self.file_path, 'r') as f:

            self.progress.emit(10)

            data['dff_traces'] = [f['dff'][i, :] for i in range(f['dff'].shape[0])]
            self.progress.emit(30)

            data['mask_data'] = f['A'][:]
            data['fov'] = f['Cn'][:]
            data['raw_cal'] = f['C'][:]
            self.progress.emit(50)

            num_pixels, num_rois = data['mask_data'].shape
            side = int(np.sqrt(num_pixels))
            data['masks_2d'] = [
                data['mask_data'][:, i].reshape(side, side, order='F')
                for i in range(num_rois)
            ]
            self.progress.emit(70)

            data['accepted'] = f['idx_components'][:]
            data['rejected'] = f['idx_components_bad'][:]

        # find tiff files
        parent_dir = os.path.dirname(os.path.dirname(self.file_path))
        tiff_dir = os.path.join(parent_dir, 'motion_corrected_tiffs')
        print(tiff_dir)
        tiff_files = glob.glob(os.path.join(tiff_dir, '*.tif'))
        # find the length of the first tiff file
        with tifffile.TiffFile(tiff_files[0]) as tif:
            num_frames = len(tif.pages)
        data['tiff_files'] = tiff_files
        data['frames_per_file'] = num_frames
        self.progress.emit(100)

        self.finished.emit(data)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Layout Example")
        self.resize(1200, 800)
        self.dff_traces = None
        self.mask_data = None
        self.masks_2d = None
        self.fov = None
        self.accept = None
        self.reject = None
        self.raw_cal = None
        self.current_roi = 0

        self.video_frames = None   # numpy array of frames to play
        self.video_index = 0
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.play_video_frame)
        self.video_margin = 10   
        self.video_scale = 5
        self.video_timer.setInterval(50)

        self.video_trace_img = None   # OpenCV image for trace
        self.video_trace_label = None # QLabel to display trace
        self.current_video_indices = None

        self.init_ui()


    def init_ui(self):
        # === Top Buttons ===
        btn_load = QPushButton("Load File")
        btn_load.clicked.connect(self.load_file) 
        
        btn_next = QPushButton("Next ROI")
        btn_next.clicked.connect(self.next_roi)
        btn_prev = QPushButton("Prev ROI")
        btn_prev.clicked.connect(self.prev_roi)

        btn_accept = QPushButton("Accept")
        btn_reject = QPushButton("Reject")
        btn_save = QPushButton("Save ROIs")

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_load)
        button_layout.addWidget(btn_prev)
        button_layout.addWidget(btn_next)
        button_layout.addWidget(btn_accept)
        button_layout.addWidget(btn_reject)
        button_layout.addWidget(btn_save)
        
        # === Left Panel (figure and small plot) ===
        self.figure_panel = QFrame()
        self.figure_panel.setFrameShape(QFrame.Shape.Box)
        self.figure_panel.setStyleSheet("background-color: lightgray")

        self.figure_label = HoverableLabel()
        self.figure_label.roi_clicked.connect(self.on_roi_clicked)
        self.figure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.figure_label.setScaledContents(True)
        figure_layout = QVBoxLayout()
        #figure_layout.addWidget(self.figure_label)
        figure_layout.addWidget(self.figure_label)
        self.figure_panel.setLayout(figure_layout)

        self.small_plot_panel = QFrame()
        self.small_plot_panel.setFrameShape(QFrame.Shape.Box)
        self.small_plot_panel.setMinimumHeight(150)
        self.small_plot_panel.setStyleSheet("background-color: lightblue")

        # Add QLabel to display traces
        # Create figure and canvas
        self.trace_figure = Figure(figsize=(5, 2))
        self.trace_canvas = FigureCanvas(self.trace_figure)

        # Layout inside small_plot_panel
        trace_layout = QVBoxLayout()
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.addWidget(self.trace_canvas)
        self.small_plot_panel.setLayout(trace_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.figure_panel)
        left_layout.addWidget(self.small_plot_panel)

        # === Right Panel (video and small plot) ===
        self.video_panel = QFrame()
        self.video_panel.setFrameShape(QFrame.Shape.Box)
        self.video_panel.setMinimumWidth(300)
        self.video_panel.setMinimumHeight(400)
        self.video_panel.setStyleSheet("background-color: lightgreen")
        
        self.video_label = QLabel(self.video_panel)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_panel_layout = QVBoxLayout()
        self.video_panel_layout.setContentsMargins(0,0,0,0)
        self.video_panel_layout.addWidget(self.video_label)
        self.video_panel.setLayout(self.video_panel_layout)


        self.right_small_plot = QFrame()
        self.right_small_plot.setFrameShape(QFrame.Shape.Box)
        self.right_small_plot.setMinimumHeight(150)
        self.right_small_plot.setStyleSheet("background-color: lightyellow")

        # --- Add a QLabel to display video trace ---
        self.video_trace_label = QLabel()
        self.video_trace_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        trace_layout = QVBoxLayout()
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.addWidget(self.video_trace_label)
        self.right_small_plot.setLayout(trace_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_panel)
        right_layout.addWidget(self.right_small_plot)

        # --- Status Panel ---

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

       

        # === Main Content Layout ===
        main_content_layout = QHBoxLayout()
        main_content_layout.addLayout(left_layout)
        main_content_layout.addLayout(right_layout)

        # === Overall Layout ===
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(main_content_layout)

        main_layout.addWidget(self.progress_bar)
        
        self.setLayout(main_layout)

    def log_status(self, message: str):
        """Append a message to the status panel with automatic scrolling."""
        self.status_panel.append(message)
        self.status_panel.verticalScrollBar().setValue(self.status_panel.verticalScrollBar().maximum())

    def load_file(self):
        # Open file dialog to select HDF5 file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 File", "", "HDF5 Files (*.h5 *.hdf5)"
        )
        if not file_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.thread = QThread()
        self.worker = LoadWorker(file_path)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.loading_finished)
        self.worker.error.connect(self.loading_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def loading_finished(self, data):
        self.progress_bar.setVisible(False)

        self.dff_traces = data['dff_traces']
        self.mask_data = data['mask_data']
        self.fov = data['fov']
        self.masks_2d = data['masks_2d']
        self.accepted = data['accepted']
        self.rejected = data['rejected']
        self.raw_cal = data['raw_cal']
        self.tiff_files = data['tiff_files']
        self.frames_per_file = data['frames_per_file']
        self.current_roi = 0

        self.display_fov()


    def loading_error(self, message):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Loading Error", message)

    def display_fov(self):
        if self.fov is None or self.masks_2d is None:
            return

        # Normalize FOV to 0–255
        img = self.fov.astype(np.float32)
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype(np.uint8)

        # Convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w, ch = img_bgr.shape

        # Convert to QImage
        q_img = QImage(img_bgr.data, w, h, w * ch, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        # --- IMPORTANT PART ---
        # Reshape masks to match FOV once
        masks_reshaped = []
        for roi_mask in self.masks_2d:
            roi_mask_2d = roi_mask.reshape((h, w), order='F')
            masks_reshaped.append(roi_mask_2d)

        # Send both pixmap and masks to HoverableLabel
        self.figure_label.set_base_image(img_bgr, masks_reshaped)
        self.figure_label.set_pixmap_and_masks(pixmap, masks_reshaped)

        # Optional: adjust panel size
        self.figure_label.setFixedSize(w, h)
        self.figure_panel.setFixedSize(w + 10, h + 10)

    def plot_current_roi_trace(self):
        if self.raw_cal is None or self.dff_traces is None:
            print("Traces not loaded.")
            return

        # Use the currently selected ROI
        roi_index = self.current_roi
        if roi_index is None or roi_index >= len(self.raw_cal):
            roi_index = 0  # fallback to first ROI if not set

        raw = self.raw_cal[roi_index]
        dff = self.dff_traces[roi_index]

        # Clear previous figure
        self.trace_figure.clear()

        # Dual y-axis
        ax1 = self.trace_figure.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(raw, color='blue', linewidth=1, label='Raw')
        ax2.plot(dff, color='red', linewidth=1, label='ΔF/F')

        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Raw Fluorescence", color='blue')
        ax2.set_ylabel("ΔF/F", color='red')

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Draw the figure
        self.trace_canvas.draw()

    def on_roi_clicked(self, roi_index):
        self.current_roi = roi_index            # MainWindow tracks selected ROI
        self.figure_label.current_roi = roi_index  # Tell label which ROI is selected
        self.display_fov()                       # redraw FOV
        self.plot_current_roi_trace()   
        self.load_roi_video()

    def load_roi_video(self):

        if self.masks_2d is None or self.fov is None:
            return

        roi_mask = self.masks_2d[self.current_roi]
        h, w = int(np.sqrt(roi_mask.size)), int(np.sqrt(roi_mask.size))
        roi_mask_2d = roi_mask.reshape((h, w), order='F')

        # Find bounding box of ROI
        ys, xs = np.where(roi_mask_2d > 0)
        y_min, y_max = max(0, ys.min() - 50), min(h, ys.max() + 50)  # add small padding
        x_min, x_max = max(0, xs.min() - 50), min(w, xs.max() + 50)

        # Load TIFF files

        # Use dF/F or raw to find most active consecutive 600 frames
        roi_trace = self.dff_traces[self.current_roi]
        window = 600
        start_idx = np.argmax(np.convolve(roi_trace, np.ones(window), mode='valid'))

        frame_indices = np.arange(start_idx, start_idx + window)

        # Load frames and crop to ROI patch
        first_file_idx = frame_indices[0] // self.frames_per_file
        last_file_idx = frame_indices[-1] // self.frames_per_file
        self.video_frames = []
        frame_count = first_file_idx * self.frames_per_file

        for tiff_idx in range(first_file_idx, last_file_idx + 1):
            #self.log_status(f"Loading TIFF file: {self.tiff_files[tiff_idx]}......")
            tiff_file = self.tiff_files[tiff_idx]
            imgs = tifffile.imread(tiff_file)  # shape: (num_frames, H, W)
            for i in range(imgs.shape[0]):
                if frame_count in frame_indices:
                    patch = imgs[i, y_min:y_max+1, x_min:x_max+1]
                    self.video_frames.append(patch)
                frame_count += 1

        self.video_frames = np.array(self.video_frames, dtype=np.uint16)
        self.video_index = 0

        #self.log_status(f"Loaded {len(self.video_frames)} frames for ROI {self.current_roi}")

        if len(self.video_frames) > 0:
            self.video_timer.start()   # <<< start the playback timer
        #else:
            #self.log_status("No frames to play for this ROI.")

        self.create_video_trace_image()

    def play_video_frame(self):
        if not hasattr(self, 'video_frames') or len(self.video_frames) == 0:
            return

        frame = self.video_frames[self.video_index]  # this is already cropped patch
        h, w = frame.shape

        # Scale frame to 0-255
        frame_scaled = frame.astype(np.float32)
        frame_scaled = 255 * (frame_scaled - frame_scaled.min()) / (frame_scaled.max() - frame_scaled.min() + 1e-8)
        frame_scaled = frame_scaled.astype(np.uint8)

        scale = self.video_scale
        scaled_h = h * scale
        scaled_w = w * scale
        frame_scaled = cv2.resize(frame_scaled, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        # Convert to BGR for contours
        frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_GRAY2BGR)

        if self.masks_2d is not None:
            # Bounding box of the ROI in original FOV
            roi_mask = self.masks_2d[self.current_roi].reshape(int(np.sqrt(self.masks_2d[self.current_roi].size)),
                                                                int(np.sqrt(self.masks_2d[self.current_roi].size)), order='F')
            ys, xs = np.where(roi_mask > 0)
            y_min, y_max = max(0, ys.min() - 5), min(roi_mask.shape[0], ys.max() + 5)
            x_min, x_max = max(0, xs.min() - 5), min(roi_mask.shape[1], xs.max() + 5)

            # --- Selected ROI contour ---
            roi_patch = (roi_mask[y_min:y_max, x_min:x_max] > 0).astype(np.uint8) * 255
            roi_patch_scaled = cv2.resize(roi_patch, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(roi_patch_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(frame_bgr, [cnt], -1, (0, 0, 255), 1)  # red

            # --- Other ROIs ---
            for i, mask in enumerate(self.masks_2d):
                if i == self.current_roi:
                    continue
                mask_2d = mask.reshape(int(np.sqrt(mask.size)), int(np.sqrt(mask.size)), order='F')
                mask_patch = (mask_2d[y_min:y_max, x_min:x_max] > 0).astype(np.uint8) * 255
                if mask_patch.sum() == 0:
                    continue
                mask_patch_scaled = cv2.resize(mask_patch, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_patch_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(frame_bgr, [cnt], -1, (0, 255, 0), 1)  # green

        # Convert to QImage and display
        q_img = QImage(frame_bgr.data, scaled_w, scaled_h, scaled_w*3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

        # Advance frame
        self.video_index = (self.video_index + 1) % len(self.video_frames)

        # Update trace vertical line
        if self.video_trace_img is not None and self.current_video_indices is not None:
            self.update_video_trace_line(self.video_index)


    def create_video_trace_image(self):
        """Prepare ΔF/F trace for progressive plotting with video."""
        if self.dff_traces is None or self.current_roi is None or self.video_frames is None:
            return

        roi_trace = self.dff_traces[self.current_roi]
        window = len(self.video_frames)
        start_idx = np.argmax(np.convolve(roi_trace, np.ones(window), mode='valid'))
        frame_indices = np.arange(start_idx, start_idx + window)
        self.current_video_indices = frame_indices

        # Normalize trace to float64 to avoid overflow
        trace_vals = roi_trace[frame_indices].astype(np.float64)
        trace_norm = 255 * (trace_vals - trace_vals.min()) / (trace_vals.max() - trace_vals.min() + 1e-8)
        self.video_trace_norm = trace_norm  # store for progressive plotting

        # Create blank white OpenCV image
        h_trace = 150
        w_trace = len(frame_indices)
        self.video_trace_img = np.ones((h_trace, w_trace), dtype=np.uint8) * 255
        self.h_trace = h_trace
        self.w_trace = w_trace

        # Setup QLabel if not exist
        if self.video_trace_label is None:
            self.video_trace_label = QLabel(self.right_small_plot)
            self.video_trace_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            layout = QVBoxLayout()
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self.video_trace_label)
            self.right_small_plot.setLayout(layout)

        # Display initial frame
        self.update_video_trace_line(0)


    def update_video_trace_line(self, frame_idx):
        """Draw ΔF/F trace up to current video frame with vertical red line."""
        if self.video_trace_img is None or self.video_trace_norm is None:
            return

        # Start with blank white image
        img = np.ones((self.h_trace, self.w_trace), dtype=np.uint8) * 255

        # Draw the trace up to current frame
        for i in range(1, frame_idx + 1):
            if i >= len(self.video_trace_norm):
                break
            y1 = int(self.h_trace - self.video_trace_norm[i-1] * self.h_trace / 255)
            y2 = int(self.h_trace - self.video_trace_norm[i] * self.h_trace / 255)
            cv2.line(img, (i-1, y1), (i, y2), 0, 1)  # black line

        # Convert to BGR for vertical red line
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = frame_idx
        if x < img_bgr.shape[1]:
            cv2.line(img_bgr, (x,0), (x,self.h_trace-1), (0,0,255), 1)

        # Convert to QImage and display
        pixmap = QPixmap.fromImage(QImage(img_bgr.data, img_bgr.shape[1],
                                        img_bgr.shape[0], img_bgr.shape[1]*3,
                                        QImage.Format.Format_BGR888))
        self.video_trace_label.setPixmap(pixmap)


    def next_roi(self):
        if self.masks_2d is not None:
            self.current_roi = (self.current_roi + 1) % len(self.masks_2d)
            self.display_fov()

    def prev_roi(self):
        if self.masks_2d is not None:
            self.current_roi = (self.current_roi - 1) % len(self.masks_2d)
            self.display_fov()

class HoverableLabel(QLabel):
    roi_clicked = pyqtSignal(int) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # important to track mouse without clicks
        self.img_pixmap = None
        self.masks_2d = []  # list of 2D ROI masks matching image
        self.hovered_roi = None  # index of ROI under mouse
        self.base_img = None
        self.masks = None
        self.current_highlight = None
        self.setMouseTracking(True)
        self.current_roi = None 

    def set_base_image(self, img_np, masks):
        self.base_img = img_np.copy()
        self.masks = masks
        self.current_highlight = None
        self.update_display()

    def set_pixmap_and_masks(self, pixmap, masks):
        self.img_pixmap = pixmap
        self.masks_2d = masks
        self.hovered_roi = None
        self.setPixmap(pixmap)

    def mouseMoveEvent(self, event):
        if self.base_img is None:
            return

        x = int(event.position().x())
        y = int(event.position().y())

        if x < 0 or y < 0 or x >= self.base_img.shape[1] or y >= self.base_img.shape[0]:
            return

        found = None
        for i, mask in enumerate(self.masks):
            if mask[y, x] > 0:
                found = i
                break

        if found != self.current_highlight:
            self.current_highlight = found
            self.update_display()

    def mousePressEvent(self, event):
        if self.masks is None or self.current_highlight is None:
            return
        self.roi_clicked.emit(self.current_highlight)

    def update_display(self):
        if self.base_img is None:
            return

        img = self.base_img.copy()
        overlay = np.zeros_like(img)

        for i, mask in enumerate(self.masks):
            if np.all(mask == 0):
                continue

            binary = (mask > 0)

            # Determine color
            if self.current_roi == i:
                color = (0, 0, 255)       # red = selected ROI
            elif self.current_highlight == i:
                color = (0, 255, 255)     # yellow = hovered ROI
            else:
                color = (0, 255, 0)       # green = other ROIs

            overlay[binary] = color

        # Blend overlay with base image
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Convert to QImage and display
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, w * ch, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
