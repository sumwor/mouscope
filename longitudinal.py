#%% analyze longitudinal registered calcium imaging data
from tokenize import group

import pandas as pd
import numpy as np
import os
import glob
import h5py
from scipy.io import loadmat
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from utils_imaging import load_h5_item, d_prime
import pickle


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

        # not tracking every session, 
        # odor (AB1, (last AB before tansition) CD1, CD3)
        # rotarod (trial 1/2, trial 6/7, trial 12) 

        self.data_index =  pd.concat([self.rotarod.data_index, self.Odor.data_index], ignore_index=True)
        self.data_index['Task'] = ['Rotarod']*len(self.rotarod.data_index) + ['Odor']*len(self.Odor.data_index) 

        # go over each animal, look for the longitudinal registered data, and for each session, find the referecence field of view
        self.Animal = self.data_index['Animal'].unique() 
        self.long_reg = {}  # dictionary to save the longitudinal registered data. key : animal names
        self.ses_included = {}
        self.ses_included_long = {}

        self.data_index['long_session'] = None
        self.data_index['FOV'] = None
        for aidx, aa in enumerate(self.Animal):
            long_data_path =  glob.glob(os.path.join(self.data, aa, 'Longitudinal', '*.mat'))

            savefigpath = os.path.join(self.analysis, self.data_index['Animal'][0], 'Longitudinal')
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)
            # load longitudinal registered data
            
            long_data = {}

            # HDF5 signature = MATLAB v7.3
            with h5py.File(long_data_path[0], 'r') as f:
                group = f['cell_registered_struct']
    
                for key in group.keys():
                    long_data[key] = load_h5_item(f, group[key])

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
                        else:
                            Cn = np.load(FOV_path_ave, allow_pickle=True)

            self.long_reg[aa] = long_data
            # in long_data, plot contours on FOV 
            # check ROI contours across session, and calcium traces
            
            #%%         
            # not tracking every session, 
            # odor (AB1, (last AB before tansition) CD1, CD3)
            # rotarod (trial 1/2, trial 6/7, trial 12) 
            #
            # look for the sessions
            # change this to trial 1 later
            self.ses_keys = ['AB1', 'AB-end', 'CD1', 'CD3', 'trial2', 'trial6', 'trial12']
            ses_included_long = {}
            ses_included = {} # save the index of sessions in data_index
            # find the index that meet this criteria
            ses_included_long['AB1'] =self.data_index['long_session'][np.where((self.data_index['Task'] == 'Odor') & 
                                          (self.data_index['Protocol'] == 'AB') & 
                                          (self.data_index['ProtocolDay']==1))[0]].values[0]-1
            ses_included['AB1'] = self.data_index.index[np.where((self.data_index['Task'] == 'Odor') & 
                                          (self.data_index['Protocol'] == 'AB') & 
                                          (self.data_index['ProtocolDay']==1))[0]].values[0]
            maxAB = np.max(self.data_index['ProtocolDay'][(self.data_index['Task'] == 'Odor') & 
                                                          (self.data_index['Protocol'] == 'AB')])
            ses_included_long['AB-end'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Odor') &
                                              (self.data_index['Protocol'] == 'AB') &
                                              (self.data_index['ProtocolDay'] == maxAB))[0]].values[0]-1
            ses_included['AB-end'] = self.data_index.index[np.where((self.data_index['Task'] == 'Odor') &
                                              (self.data_index['Protocol'] == 'AB') &
                                              (self.data_index['ProtocolDay'] == maxAB))[0]].values[0]
            ses_included_long['CD1'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Odor') &
                                            (self.data_index['Protocol'] == 'AB-CD') &
                                            (self.data_index['ProtocolDay'] == 1))[0]].values[0]-1
            ses_included['CD1'] = self.data_index.index[np.where((self.data_index['Task'] == 'Odor') &
                                            (self.data_index['Protocol'] == 'AB-CD') &
                                            (self.data_index['ProtocolDay'] == 1))[0]].values[0]
            ses_included_long['CD3'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Odor') &
                                            (self.data_index['Protocol'] == 'AB-CD') &
                                            (self.data_index['ProtocolDay'] == 3))[0]].values[0]-1
            ses_included['CD3'] = self.data_index.index[np.where((self.data_index['Task'] == 'Odor') &
                                            (self.data_index['Protocol'] == 'AB-CD') &
                                            (self.data_index['ProtocolDay'] == 3))[0]].values[0]
            ses_included_long['trial2'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 2))[0]].values[0]-1
            ses_included['trial2'] = self.data_index.index[np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 2))[0]].values[0]
            ses_included_long['trial6'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 6))[0]].values[0]-1
            ses_included['trial6'] = self.data_index.index[np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 6))[0]].values[0]
            ses_included_long['trial12'] = self.data_index['long_session'][np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 12))[0]].values[0]-1
            ses_included['trial12'] = self.data_index.index[np.where((self.data_index['Task'] == 'Rotarod') &
                                                (self.data_index['Trial'] == 12))[0]].values[0]
            self.ses_included_long[aa] = ses_included_long # save the index of sessions (in long_data
            # since there might be missing trials, so index in the long_data might be differ from the data_index)
            self.ses_included[aa] = ses_included # save the index of sessions in data_index
            # convert dict to list
            ses_included_list = [ses_included[sk] for sk in self.ses_keys]
            # look for co-registerd neurons in these sessions
            coregistered_cells = long_data['cell_to_index_map'] 
            # shape: (nSess, nNeurons), value: neuron index in each session, 0 if not exist

            #%% make three plot:
            # put it here for now, will become an average with more animals
            # 1. coregistered cells across odor and rotarod
            # 2. coregistered cells in odor
            # 3. coregistered cells i rotarod
            savefigname = os.path.join(savefigpath, 'Register_overview.png')
            if not os.path.exists(savefigname): 
            
                plt.figure()
                plt.tight_layout()
                # find the neuron index that >0 in all sessions
                plt.subplot(2,2,1)
                plt.imshow(Cn)
                plt.title('Coregistered neurons all')
                neuron_all= np.where(np.all(coregistered_cells[ses_included_list,:]>0, axis=0))[0]
                # find the index of these neurons in session 1
                shared_session1 = coregistered_cells[0, neuron_all].astype(int)-1
                shared_contour_s1 = long_data['spatial_footprints_corrected'][0][:,:,shared_session1]
                for nn in range(len(neuron_all)):
                    img = shared_contour_s1[:, :, nn]
                    cy, cx = center_of_mass(img) # get the center of mass
                                    
                    thr = np.percentile(shared_contour_s1[:,:, nn], 70)
                    plt.contour(shared_contour_s1[:,:,nn], levels=[thr])
                    plt.scatter(cx, cy, c='r')  # show COM

                plt.subplot(2,2,2)
                # coregistered neurons in odor task
                plt.imshow(Cn)
                plt.title('Coregistered neurons odor')
                neuron_odor= np.where(np.all(coregistered_cells[ses_included_list[0:4],:]>0, axis=0))[0]
                # find the index of these neurons in odor session 1
                shared_session1 = coregistered_cells[ses_included_list[0], neuron_odor].astype(int)-1
                shared_contour_s1 = long_data['spatial_footprints_corrected'][ses_included_list[0]][:,:,shared_session1]
                for nn in range(len(neuron_odor)):
                    img = shared_contour_s1[:, :, nn]
                    cy, cx = center_of_mass(img) # get the center of mass
                                    
                    thr = np.percentile(shared_contour_s1[:,:, nn], 70)
                    plt.contour(shared_contour_s1[:,:,nn], levels=[thr])
                    plt.scatter(cx, cy, c='r')  # show COM

                # coregistered neurons in rotarod
                plt.subplot(2,2,3)
                plt.imshow(Cn)
                plt.title('Coregistered neurons rotarod')
                neuron_rotarod= np.where(np.all(coregistered_cells[ses_included_list[4:],:]>0, axis=0))[0]
                # find the index of these neurons in rotarod session 
                shared_session1 = coregistered_cells[ses_included_list[4], neuron_rotarod].astype(int)-1
                shared_contour_s1 = long_data['spatial_footprints_corrected'][ses_included_list[4]][:,:,shared_session1]
                for nn in range(len(neuron_rotarod)):
                    img = shared_contour_s1[:, :, nn]
                    cy, cx = center_of_mass(img) # get the center of mass
                                    
                    thr = np.percentile(shared_contour_s1[:,:, nn], 70)
                    plt.contour(shared_contour_s1[:,:,nn], levels=[thr])
                    plt.scatter(cx, cy, c='r')  # show COM
                #

                plt.subplot(2,2,4)
                # plot percentage of neurons are coregistered in both, odor and rotarod

                # Sessions
                n_sessions = len(ses_included_list)
                

                # Precompute neuron sets
                all_neurons = np.where(np.all(coregistered_cells[ses_included_list, :] > 0, axis=0))[0]
                odor_neurons = np.where(np.all(coregistered_cells[ses_included_list[0:4], :] > 0, axis=0))[0]
                rotarod_neurons = np.where(np.all(coregistered_cells[ses_included_list[4:], :] > 0, axis=0))[0]

                # Prepare percentages for each session
                percent_all = []
                percent_odor_or_rotarod = []
                percent_not = []

                for i, ses in enumerate(ses_included_list):
                    # All-session neurons fraction
                    n_all = len(neuron_all)  # coregistered all
                    total_neurons = np.sum(coregistered_cells[ses]>0) # total neuron registered
                    
                    if i < 4:  # odor sessions
                        n_odor = len(odor_neurons)
                        n_not = total_neurons - n_odor
                        percent_all.append(n_all / total_neurons * 100)
                        percent_odor_or_rotarod.append((len(odor_neurons) - n_all)/total_neurons*100)
                        percent_not.append((total_neurons - n_odor)/total_neurons*100)
                        
                    else:  # rotarod sessions
                        n_rot = len(rotarod_neurons)
                        n_not = total_neurons - n_rot
                        percent_all.append(n_all / total_neurons * 100)
                        percent_odor_or_rotarod.append((len(rotarod_neurons) - n_all)/total_neurons*100)
                        percent_not.append((total_neurons - n_rot)/total_neurons*100)
                        

                # Convert to numpy arrays
                percent_all = np.array(percent_all)
                percent_odor_or_rotarod = np.array(percent_odor_or_rotarod)
                percent_not = np.array(percent_not)

                # Plot stacked bars
                x = np.arange(n_sessions)

                plt.bar(x, percent_all, label='All sessions')
                plt.bar(x, percent_odor_or_rotarod, bottom=percent_all,
                        label='Odor / Rotarod only')
                plt.bar(x, percent_not, bottom=percent_all + percent_odor_or_rotarod,
                        label='Not coregistered')

                plt.xticks(x, [ses for ses in ses_keys], rotation=45)
                plt.ylabel('Percentage of neurons (%)')
                plt.title('Neuron coregistration per session')
                plt.legend()

                plt.savefig()
                plt.close()

    def auROC_coregister(self):

        # load auROC result, track it across sessions
        # look for fraction of auROC neurons in the coregistered neurons, and neurons not coresigered
            for aidx, aa in enumerate(self.Animal):
                long_data = self.long_reg[aa]
                ses_included_long = self.ses_included_long[aa]
                ses_included = self.ses_included[aa]
                # shape: (nSess, nNeurons), value: neuron index in each session, 0 if not exist
                odor_ses = self.ses_keys[0:4]
                nFiles = len(odor_ses)


                # temp varible for plot
                # track the percentage of neurons in coregisterd pop and non-coregistered pop that are significant in auROC
                n_sig_AB = np.full((2, nFiles), np.nan)
                n_sig_CD = np.full((2, nFiles), np.nan)
                perf_AB = np.full((1, nFiles), np.nan)
                perf_CD = np.full((1, nFiles), np.nan)
                d_AB = np.full((1, nFiles), np.nan)
                d_CD = np.full((1, nFiles), np.nan)
                crit_AB = np.full((1, nFiles), np.nan)
                crit_CD = np.full((1, nFiles), np.nan) # criterion for d prime (bias)

                for ii in range(nFiles):
                    ses_idx= ses_included[odor_ses[ii]]
                    savedatapath = os.path.join(self.analysis, self.data_index['Animal'][ses_idx], 'Odor', 'Imaging',
                            self.data_index['Date'][ses_idx], 'Result')
                    behDF = pd.read_csv(self.data_index['BehCSV'][ses_idx])
                    perf_AB[0,ii] = np.nanmean(behDF['reward'][(behDF['schedule']==1) | (behDF['schedule']==2)]>0)
                    perf_CD[0,ii] = np.nanmean(behDF['reward'][(behDF['schedule']==3) | (behDF['schedule']==4)]>0)
                    
                    # calculate d_prime
                    d_data_AB = {}
                    d_data_AB['stimulus'] = np.array(behDF['schedule'][(behDF['schedule']==1) | (behDF['schedule']==2)]).astype(int)
                    d_data_AB['actions'] = np.array(behDF['actions'][(behDF['schedule']==1) | (behDF['schedule']==2)]).astype(int)+1
                    [d_AB[0,ii], crit_AB[0,ii]] = d_prime(d_data_AB)
                    if self.data_index['Protocol'][ses_idx] == 'AB-CD':
                        d_data_CD = {}
                        d_data_CD['stimulus'] = np.array(behDF['schedule'][(behDF['schedule']==3) | (behDF['schedule']==4)]).astype(int)-2
                        d_data_CD['actions'] = np.array(behDF['actions'][(behDF['schedule']==3) | (behDF['schedule']==4)]).astype(int)+1
                        [d_CD[0,ii], crit_CD[0,ii]] = d_prime(d_data_CD)

                    savedataname = os.path.join(savedatapath, 
                                                f"auROC_results_stimulus_{self.data_index['Protocol'][ii]}.pkl")