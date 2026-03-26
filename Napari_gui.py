# a Napari gui that allows the user to select a hdf5 file
# and then visualize the data in it. The user can also select a region of interest (ROI)
import sys
import h5py
import napari
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QSizePolicy    
)
import numpy as np

class NapariGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Napari GUI")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        
    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.create_widgets()
        self.create_layout()

    def create_widgets(self):
        self.h5_button = QPushButton("Select HDF5 File")
        self.h5_button.clicked.connect(self.select_h5_file)
        self.layout.addWidget(self.h5_button)
        self.viewer = napari.Viewer()
        self.layout.addWidget(self.viewer.window.qt_viewer)

    def create_layout(self):
        self.h5_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.viewer.window.qt_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def select_h5_file(self):
        h5_path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File", "", "HDF5 Files (*.h5 *.hdf5)")
        if h5_path:
            self.load_h5_data(h5_path)

    def load_h5_data(self, h5_path):
        # Load data from HDF5 file and visualize in Napari
        with h5py.File(h5_path, 'r') as f:
            # Assuming the dataset is stored under the key 'data'
            dff_traces = [f['dff'][i, :] for i in range(f['dff'].shape[0])]
            mask_data = f['A']
            num_pixels, num_rois = mask_data.shape
            side = int(np.sqrt(num_pixels))
            masks_2d = [mask_data[:, i].reshape(side, side, order='F') for i in range(num_rois)]
            accepted = f['idx_components'][:]
            rejected = f['idx_components_bad'][:]
            fov_corr = f['Cn']
            self.viewer.add_image(fov_corr, name="Field of View")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NapariGUI()
    window.show()
    sys.exit(app.exec())