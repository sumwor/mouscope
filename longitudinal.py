#%% analyze longitudinal registered calcium imaging data
import pandas as pd
import numpy as np
import os
import glob
import h5py
from scipy.io import loadmat

class Longitudinal:
    def __init__(self, root_dir, Odor, rotarod):

        # take odor and rotarod imaging object as input
        self.root_dir = root_dir
        self.Odor = Odor
        self.rotarod = rotarod
        self.data = Odor.data
        self.analysis = Odor.analysis
        self.summary = Odor.summary

        self.load_data()


    def load_data(self):
        # load the longitudinal registered data, including the registered images, the extracted df/f traces, and the aligned behavior events
        # go through each animal, find the longitudinal registered data, and for each session, find the referecence field of view
        # reference FOV: std projection and correlation FOV

        # concatenate data_index from odor and rotarod together, add one column to indicate the task
        self.data_index =  pd.concat([self.rotarod.data_index, self.Odor.data_index], ignore_index=True)
        self.data_index['Task'] = ['Rotarod']*len(self.rotarod.data_index) + ['Odor']*len(self.Odor.data_index) 

        # go over each animal, look for the longitudinal registered data, and for each session, find the referecence field of view
        self.Animal = self.data_index['Animal'].unique() 

        self.data_index['long_session'] = None
        self.data_index['FOV'] = None
        for aidx, aa in enumerate(self.Animal):
            long_data_path =  glob.glob(os.path.join(self.data, aa, 'Longitudinal', '*.mat'))
            # load longitudinal registered data
            
            long_data = {}

            # HDF5 signature = MATLAB v7.3
            with h5py.File(long_data_path[0], 'r') as f:
                # show all keys
                for key in f['cell_registered_struct'].keys():
                    long_data[key] = f['cell_registered_struct'][key][()]  

            # 'cell_to_index_map' is the matrix that registers cells across sessions
            # rotarod sessions - then odor sessions
            # map session number to data_index for ifCalImg=True sessions
            nSess = 1
            for ss in range(self.data_index.shape[0]):
                if self.data_index['Animal'][ss] == aa:
                    if self.data_index['ifCalImg'][ss]:
                        self.data_index.loc[ss,'long_session'] = nSess  # session count in longitudinal registration
                        nSess += 1

                        # load reference frame
                        if self.data_index['Task'][ss]=='Rotarod':
                            FOV_path_ave = os.path.join(self.data, aa, self.data_index['Task'][ss], 'Imaging',
                                                    f"{aa}_{self.data_index['Date'][ss]}", 
                                                    'caiman_results', f'trial{int(self.data_index["Trial"][ss])}_correlation.npy')
                        elif self.data_index['Task'][ss]=='Odor':
                            # get the correlation plot from caiman as reference
                            FOV_path_ave = os.path.join(self.data, aa, self.data_index['Task'][ss], 'Imaging',
                                                    f"{aa}_{self.data_index['Date'][ss]}", 
                                                    'caiman_results', f'{self.data_index["Date"][ss]}_correlation.npy')

                        # if file not exist, load dFFFile, extract Cn (correlation FOV), save it for reference
                        self.data_index.loc[ss,'FOV'] = FOV_path_ave
                        if not os.path.exists(FOV_path_ave):
                            # load dFFFile
                            dFFFile = self.data_index['dFFFile'][ss]
                            with open(dFFFile, 'rb') as f:
                                header = f.read(8)

                            # HDF5 signature = MATLAB v7.3
                            if header.startswith(b'MATLAB 7'):
                                with h5py.File(dFFFile, "r") as f:
                                    Cn = f['results']['Cn'][()]
                            else:
                                data_temp = loadmat(dFFFile)
                                Cn = data_temp['results']['Cn']


                            # save Cn to the file
                            np.save(FOV_path_ave, Cn)

            # in long_data, plot contours on FOV 
            # check ROI contours across session, and calcium traces
            
            # 