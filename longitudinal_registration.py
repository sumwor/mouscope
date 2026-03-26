# multiple session registration
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pickle
import os 
import glob
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import caiman as cm
from scipy.io import loadmat
from scipy.sparse import csc_matrix

import h5py

import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') 
plt.ion()

import matlab.engine
eng = matlab.engine.start_matlab()

# Load multisession data (spatial components and mean intensity templates) (should be replaced by actual data)

# spatial: a list of ROIs spatial footprints for each session
# templates: a list of mean intensity images for each session
root_dir = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001'

cnmfe_results = []
template_files = []

rotarod_folder = os.path.join(root_dir, 'Rotarod', 'Imaging')
odor_folder = os.path.join(root_dir, 'Odor', 'Imaging')

date_folders_rotarod = sorted([f for f in os.listdir(rotarod_folder) if os.path.isdir(os.path.join(rotarod_folder, f))])
date_folders_odor = sorted([f for f in os.listdir(odor_folder) if os.path.isdir(os.path.join(odor_folder, f))])

for date in date_folders_rotarod:
    caiman_folder = os.path.join(rotarod_folder, date, 'caiman_results')
    if os.path.exists(caiman_folder):
        cnmfe_file = glob.glob(os.path.join(caiman_folder, '*_updated_cnmf.mat'))
    template_folder = os.path.join(rotarod_folder, date, 'temp')
    if os.path.exists(template_folder):
        template_file = glob.glob(os.path.join(template_folder, '*_0000_d1_300_d2_300_d3_1*.mmap'))

    cnmfe_results.extend(cnmfe_file)
    template_files.extend(template_file)

    
for date in date_folders_odor:
    caiman_folder = os.path.join(odor_folder, date, 'caiman_results')
    if os.path.exists(caiman_folder):
        cnmfe_file = glob.glob(os.path.join(caiman_folder, '*updated_cnmf.mat'))
    template_folder = os.path.join(odor_folder, date, 'temp')
    if os.path.exists(template_folder):
        template_file = glob.glob(os.path.join(template_folder, '*_0000_d1_300_d2_300_d3_1*.mmap'))

    cnmfe_results.extend(cnmfe_file)
    template_files.extend(template_file)

nFiles = len(cnmfe_results)

spatial = []
templates = []
spatial_reshaped = []

for i in range(nFiles):
    # check .mat version
        # check header
    with open(cnmfe_results[i], 'rb') as f:
        header = f.read(128)
    if b'MATLAB 7.3' in header:
        with h5py.File(cnmfe_results[i], 'r') as f:
            A_grp = f['results']['A']
            #print(len(A_grp['jc']))
            #shape = A_grp['shape'][()]
            A = csc_matrix((
                A_grp['data'][()],
                A_grp['ir'][()],
                A_grp['jc'][()]
            ), shape = (90000, len(A_grp['jc']) - 1))
            
            #print(A.shape)
            spatial.append(A)

            n_neurons = A.shape[1]
            A_dense = A.toarray()
            A_reshaped = A_dense.T.reshape((n_neurons, 300, 300))
            spatial_reshaped.append(A_reshaped)
    else:
        cnmfe_data = loadmat(cnmfe_results[i])
        A = cnmfe_data['results']['A'][0][0]
        spatial.append(A)

        n_neurons = A.shape[1]
        A_dense = A.toarray()
        A_reshaped = A_dense.T.reshape((n_neurons, 300, 300))
        spatial_reshaped.append(A_reshaped)

    savefigpath = os.path.join('Y:\HongliWang\Miniscope\ASD\Data\ASDC001\spatial_footprint', 'session'+ str(i) + '_spatial_footprint.mat')
    sio.savemat(savefigpath, {'footprints': A_reshaped})

    Yr, dims, T = cm.load_memmap(template_files[i])
    images = Yr.T.reshape((T,) + dims, order='F')
    template = np.mean(images,axis=0)
    templates.append(template)

# save spatial_reshaped to a .mat file
import scipy.io as sio
import numpy as np

# spatial_reshaped is a list of arrays: each array is (n_neurons, height, width)
# Let's save them as a cell array in MATLAB

mat_dict = {}
for i, arr in enumerate(spatial_reshaped):
    # Convert each array to object so it becomes a MATLAB cell
    mat_dict[f'session_{i+1}'] = arr

# Save to .mat file
sio.savemat('Y:\HongliWang\Miniscope\ASD\Data\ASDC001\spatial_reshaped.mat', mat_dict)

dims = templates[0].shape

spatial_union, assignments, matchings = register_multisession(A=spatial[0:10], dims=dims, templates=templates[0:10], max_thr=0.8, thresh_cost=1)

n_reg = 7 # minimal number of sessions that each component has to be registered in

# Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);

# Use filtered indices to select the corresponding spatial components
spatial_filtered = spatial[0][:, assignments_filtered[:, 0]]

# Plot spatial components of the selected components on the template of the last session
visualization.plot_contours(spatial_filtered, templates[-1])

visualization.plot_contours(spatial_reshaped[0], templates[-1])