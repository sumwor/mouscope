import sys
import os
import re
import glob
import numpy as np
import h5py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import napari
from scipy.spatial import ConvexHull

# -----------------------------
# Memmap wrapper
# -----------------------------
class MemmapVideo:
    def __init__(self, mmap, H, W):
        self.mmap = mmap
        self.H = H
        self.W = W
        self.total_frames = mmap.shape[0]

    def __getitem__(self, idx):
        # idx can be int or slice
        return self.mmap[idx]

# -----------------------------
# Main GUI class
# -----------------------------
class NeuronReviewer(QMainWindow):
    def __init__(self, video_stack, masks_2d, dff_traces):
        super().__init__()
        self.setWindowTitle("Neuron Reviewer (PyQt6 + Napari)")
        self.video_stack = video_stack
        self.masks_2d = masks_2d
        self.dff_traces = dff_traces

        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Napari viewer
        self.viewer = napari.Viewer(ndisplay=2, show=False)
        self.viewer.window.qt_viewer.setParent(None)
        self.viewer_widget = self.viewer.window.qt_viewer
        main_layout.addWidget(self.viewer_widget)

        # Show first 100 frames (Napari requires a loadable array)
        init_frames = min(100, self.video_stack.total_frames)
        self.image_layer = self.viewer.add_image(self.video_stack[:init_frames], name="Video", contrast_limits=[0, 5000])

        # Add ROI shapes
        self.roi_shapes = []
        for mask in self.masks_2d:
            ys, xs = np.where(mask > 0)
            points = np.column_stack([xs, ys])
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                shape = self.viewer.add_shapes([hull_points], shape_type='polygon',
                                               edge_color='red', face_color='transparent', name='ROI')
                self.roi_shapes.append(shape)

        # Right side: Matplotlib + buttons
        side_widget = QWidget()
        side_layout = QVBoxLayout()
        side_widget.setLayout(side_layout)
        main_layout.addWidget(side_widget)

        # DF/F plot
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvas(self.fig)
        side_layout.addWidget(self.canvas)

        # Buttons
        self.btn_prev = QPushButton("Prev ROI (A)")
        self.btn_next = QPushButton("Next ROI (D)")
        self.btn_accept = QPushButton("Accept")
        self.btn_reject = QPushButton("Reject")
        side_layout.addWidget(self.btn_prev)
        side_layout.addWidget(self.btn_next)
        side_layout.addWidget(self.btn_accept)
        side_layout.addWidget(self.btn_reject)

        # Status label
        self.status = QLabel("")
        side_layout.addWidget(self.status)

        # State
        self.current_idx = 0
        self.accepted = []
        self.rejected = []

        # Connect buttons
        self.btn_next.clicked.connect(self.next_roi)
        self.btn_prev.clicked.connect(self.prev_roi)
        self.btn_accept.clicked.connect(self.accept_roi)
        self.btn_reject.clicked.connect(self.reject_roi)

        # Keyboard shortcuts
        self.viewer.bind_key('a', lambda viewer: self.prev_roi())
        self.viewer.bind_key('d', lambda viewer: self.next_roi())
        self.viewer.bind_key('space', lambda viewer: self.reject_roi())

        # Initial update
        self.update_roi()

    # -----------------------------
    # Get memmap clip for ROI
    # -----------------------------
    def get_roi_clip(self, roi_idx, n_frames=50):
        mask = self.masks_2d[roi_idx]
        ys, xs = np.where(mask > 0)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        clip = self.video_stack[:n_frames, min_y:max_y+1, min_x:max_x+1]
        return clip

    # -----------------------------
    # Update ROI display
    # -----------------------------
    def update_roi(self):
        idx = self.current_idx

        # Highlight selected ROI
        for i, shape in enumerate(self.roi_shapes):
            if i == idx:
                shape.edge_color = 'yellow'
                shape.edge_width = 3
            else:
                shape.edge_color = 'red'
                shape.edge_width = 1

        # DF/F trace
        self.ax.clear()
        self.ax.plot(self.dff_traces[idx], color='k')
        self.ax.set_title(f"ROI {idx} DF/F Trace")
        self.canvas.draw_idle()

        # Status
        self.status.setText(f"ROI {idx+1}/{len(self.masks_2d)} | "
                            f"Accepted: {len(self.accepted)}, Rejected: {len(self.rejected)}")

    def next_roi(self):
        self.current_idx = (self.current_idx + 1) % len(self.masks_2d)
        self.update_roi()

    def prev_roi(self):
        self.current_idx = (self.current_idx - 1) % len(self.masks_2d)
        self.update_roi()

    def accept_roi(self):
        idx = self.current_idx
        if idx not in self.accepted:
            self.accepted.append(idx)
        if idx in self.rejected:
            self.rejected.remove(idx)
        self.update_roi()

    def reject_roi(self):
        idx = self.current_idx
        if idx not in self.rejected:
            self.rejected.append(idx)
        if idx in self.accepted:
            self.accepted.remove(idx)
        self.update_roi()


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Ask for HDF5 file
    h5_path, _ = QFileDialog.getOpenFileName(
        None, "Select CAIMAN HDF5 file", "", "HDF5 Files (*.h5 *.hdf5)"
    )
    if not h5_path:
        QMessageBox.critical(None, "No file selected", "No HDF5 file selected. Exiting.")
        sys.exit(0)

    # Determine memmap directory
    parent_dir = os.path.dirname(os.path.dirname(h5_path))
    memmap_dir = os.path.join(parent_dir, 'temp')

    mmap_pattern = os.path.join(memmap_dir, "*_ds2__d1_300_d2_300_d3_1_order_C_frames*.mmap")
    mmap_files = glob.glob(mmap_pattern)
    if len(mmap_files) == 0:
        QMessageBox.critical(None, "No memmap file", "No memmap file found. Exiting.")
        sys.exit(0)
    mmap_files = mmap_files[0]

    match = re.search(r'C_frames_(\d+)', os.path.basename(mmap_files))
    n_frames = int(match.group(1))
    H, W = 300, 300
    mmap = np.memmap(mmap_files, dtype='uint16', mode='r', shape=(n_frames, H, W))
    video_stack = MemmapVideo(mmap, H, W)

    # Load HDF5
    with h5py.File(h5_path, 'r') as f:
        dff_traces = [f['dff'][i, :] for i in range(f['dff'].shape[0])]
        mask_data = f['A']
        num_pixels, num_rois = mask_data.shape
        side = int(np.sqrt(num_pixels))
        masks_2d = [mask_data[:, i].reshape(side, side, order='F') for i in range(num_rois)]

    # Launch GUI

    window = NeuronReviewer(video_stack, masks_2d, dff_traces)
    window.show()
    sys.exit(app.exec())
