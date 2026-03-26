import tqdm

from behavioral_pipeline import BehDataOdor
import pandas as pd
import numpy as np
import os
import glob
import shutil

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from utils_imaging import *
from scipy.sparse import csc_matrix
import h5py
import pickle
from scipy.interpolate import interp1d

from tqdm import tqdm

class Imaging:
    """
    1. Align fluorescence with behavior videos, based on behavior class (Odor and Rotorad)
    2. Align videos with behavior file (.mat)
    """
    def __init__(self, root_path, BehClass):
        # load imaging sessions based on BehClass
        self.root_path = root_path
        self.data_index = BehClass.data_index
        self.data = BehClass.data
        self.analysis = BehClass.analysis
        self.behavior = BehClass.behavior
        self.summary = BehClass.summary
        self.Animals = BehClass.Animals
        self.Genotypes = BehClass.Genotypes
        self.ImageCell = BehClass.ImageCell
        self.Hemisphere = BehClass.Hemisphere

        self.load_data()

        self.align_timeStamps()
        
    def load_data(self):
        # load related imaging/recording data and timestamp file, according to behavior file
        nFiles = self.data_index.shape[0]

        for ii in range(nFiles):
            ImagingFolder = os.path.join(self.data,self.data_index['Animal'][ii],
                                         self.behavior,'Imaging', 
                                         self.data_index['Animal'][ii]+'_'+self.data_index['Date'][ii])
            
            
            ROIFile = glob.glob(os.path.join(ImagingFolder,'caiman_result', '*.hdf5'))
            # check if ROIFile exist
            if not ROIFile:
                ifCalImg = False
            else:
                ifCalImg = True

            # if more than 1, concatenate them save to a new file, then move the original file to a separate folder "splited_session"
            behRecording = glob.glob(os.path.join(ImagingFolder,'*.mp4'))

            # check if behRecording exist
            if not behRecording:
                ifBehRecording = False
            else:
                ifBehRecording = True

            # behTimeStamp share the same name with behRecording, but is a .csv file
            behRecordingFileName = [os.path.basename(p) for p in behRecording]

            if len(behRecordingFileName) > 1:
                # get CSV for this behavior recording
                behTimeStamp = glob.glob(os.path.join(ImagingFolder, '*_behTimeStamp_concatenated.csv'))
            else:
                behTimeStamp = glob.glob(os.path.join(ImagingFolder, behRecordingFileName[0][:-4] + '.csv'))
                
                # add it to the list


            AIMatrix = glob.glob(os.path.join(ImagingFolder, '*_AITTL_*'))
            AITimsStamp = glob.glob(os.path.join(ImagingFolder, '*_AITimeStamp_*.csv'))

            ImgTimeStamp = glob.glob(os.path.join(ImagingFolder, '*_ImgTimeStamp_*.csv'))

            dFFFile = glob.glob(os.path.join(ImagingFolder,'caiman_results', 'updated_cnmf.mat'))

            # add these to the data_index
            self.data_index.loc[ii,'ROIFile'] = ROIFile[0] if ROIFile else None
            self.data_index.loc[ii,'behRecording'] = behRecording[0] if behRecording else None
            self.data_index.loc[ii,'behTimeStamp'] = behTimeStamp[0] if behTimeStamp else None
            self.data_index.loc[ii,'AIMatrix'] = AIMatrix[0] if AIMatrix else None
            self.data_index.loc[ii,'AITimsStamp'] = AITimsStamp[0] if AITimsStamp else None
            self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp[0] if ImgTimeStamp else None
            self.data_index.loc[ii,'ifCalImg'] = ifCalImg
            self.data_index.loc[ii,'ifBehRecording'] = ifBehRecording
            self.data_index.loc[ii,'dFFFile'] = dFFFile[0] if dFFFile else None


    def align_timeStamps(self):
        # files to align
        # read TTL pulse, TTL timestamp, image timestamp, behTimeStamp and align them with behavior csv file
        # to do: rotarod time alignment

        nFiles = self.data_index.shape[0]

        for ii in range(nFiles):
            # first check AI matrix and TTL pulse # in case there are breaks within a session
            AI_channels = 2
            AI_freq = 1000  # 1000 Hz frequency


            #%% align AI matrix with TTL pulse
            if os.path.exists(self.data_index['AIMatrix'][ii]):
                AI_matrix = np.fromfile(self.data_index['AIMatrix'][ii])
                AI_TimeStamp = pd.read_csv(
                        self.data_index['AITimsStamp'][ii],
                        header=None
                    ).values.squeeze()  # unit in ms
                
                ## lunghao code for AI timestamp correction
                AI_TS_interp = AI_timeStamp_correction(AI_TimeStamp)

                # rearange AI_matrix to two channels (one is ground)
                AI_matrix = AI_matrix.reshape(-1, AI_channels)
                # look for rising edges of high voltage and get the time every 3 events
                is_high = AI_matrix[:,0] > 4
                edges = np.diff(is_high.astype(int))
                rising = np.where(edges == 1)[0] + 1
                falling = np.where(edges == -1)[0] + 1
                durations = (falling - rising) / AI_freq
                # exclude durations longer than 0.2 seconds (manual valve opening)
                valid_pulses = durations < 0.2
                n_valid_events = np.sum(valid_pulses)

                # read behavior csv files
                behDF = pd.read_csv(self.data_index['BehCSV'][ii])

                # look for left correct trials
                nLeftCorrect = np.sum(np.logical_and(behDF['schedule'] == 1, behDF['reward'] > 0))
                
                # make a plot, go over behDF, if a left choice reward = 3, count 3 high voltage event
                # if a left choice reward = 2, count 2 high voltage event
                nPulses = np.sum(behDF['reward'][np.logical_or(behDF['schedule']==1, behDF['schedule']==3)])

                if not nPulses == n_valid_events:
                    print(f"Session file {self.data_index['Animal'][ii]}_{self.data_index['Date'][ii]}")
                    print("Mismatching between AI pulses and left correct trials, check!!!")

                # if match, align behDF timestamp with AI timestamp
                # make a scatter plot to show time stamp of every left correct trial aligns with each other

                LC_Mask = np.logical_and(behDF['schedule'] == 1, behDF['reward'] > 0)
                trialNumber = np.arange(behDF.shape[0])
                LC_trialNum = trialNumber[LC_Mask]

                # 
                indices = (np.concatenate(([0], np.cumsum(behDF['reward'][LC_Mask][:-1])))).astype(int)
                matched = rising[indices]

                # correct for multiple clips of the same session
                nClips = np.sum(behDF['trial']==1)
                clip_start = np.where(behDF['trial']==1)[0]

                # if nClips > 1, correct the trial time for each clip based on AI timeStamp
                LectCorrect_trialIdx = np.where((behDF['schedule'] == 1) & (behDF['reward'] > 0))[0]
                behTimeList = ['outcome','center_in', 'center_out', 'side_in', 'last_side_out']
               
                if nClips > 1:
                    t_offset_0 = AI_TS_interp[matched[0]]/1000 - behDF['side_in'][LC_trialNum[0]]
                    for cc in range(nClips-1):
                        # start from the second clip
                        clip_s = clip_start[cc+1]
                        if cc == nClips-2:
                            clip_e = behDF.shape[0]
                        else:
                            clip_e = clip_start[cc+2]-1

                        first_trial_Idx = np.where((LectCorrect_trialIdx > clip_s) & (LectCorrect_trialIdx < clip_e))[0][0]
                        AI_time = AI_TS_interp[matched[first_trial_Idx]]/1000 - t_offset_0
                        for key in behTimeList:
                            behDF.loc[clip_s:clip_e, key] += AI_time



                            
                t_offset = AI_TS_interp[matched]/1000 - behDF['outcome'][LC_trialNum]
                AI_TS_aligned = np.zeros_like(AI_TS_interp)
                # based on the offset, evenly distribute the AI_TS_interp between the trials
                for tt in range(len(behDF['outcome'][LC_trialNum])-1):
                    t0 = behDF['outcome'][LC_trialNum[tt]]
                    t1 = behDF['outcome'][LC_trialNum[tt+1]] 
                    t0_AI = AI_TS_interp[matched[tt]]/1000
                    t1_AI = AI_TS_interp[matched[tt+1]]/1000

                    if tt==0:
                        # align the time before the first left reward trial 
                        AI_tobe_aligned = AI_TS_interp[AI_TS_interp/1000 < t0_AI]/1000
                        AI_TS_aligned[AI_TS_interp/1000 < t0_AI] = AI_tobe_aligned - (t0_AI - t0)
                    elif tt == len(behDF['outcome'][LC_trialNum])-2:
                        # align the time after the last left reward trial
                        AI_tobe_aligned = AI_TS_interp[AI_TS_interp/1000 >= t1_AI]/1000
                        AI_TS_aligned[AI_TS_interp/1000 >= t1_AI] = AI_tobe_aligned - (t1_AI - t1)
                    # then align the time betwee two left reward trials
                    AI_tobe_aligned = AI_TS_interp[(AI_TS_interp/1000 >= t0_AI) & (AI_TS_interp/1000 < t1_AI)]
                    timestamps_tobe_aligned = len(AI_tobe_aligned)
                    
                    AI_TS_aligned[(AI_TS_interp/1000 >= t0_AI) & (AI_TS_interp/1000 < t1_AI)] = np.linspace(t0, t1, timestamps_tobe_aligned, endpoint=False)

            #%% based on the alignment betweeen AI_TS_interp and AI_TS_aligned, align behTimeStamp and ImgTimeStamp
            # load behavior recording timestamp if exists
            if os.path.exists(self.data_index['behTimeStamp'][ii]):
                behTimeStamp = pd.read_csv(self.data_index['behTimeStamp'][ii], header=None)
                header = ['TimeStamp']
                behTimeStamp.columns = header
                # for each timestamp in behTimeStamp, find the closest timestamp in AI_TS_interp, 
                # then replace it with the corresponding timestamp in AI_TS_aligned
                
                x = behTimeStamp['TimeStamp'].values

                idx = np.searchsorted(AI_TS_interp, x)

                # clip to valid range
                idx = np.clip(idx, 1, len(AI_TS_interp) - 1)

                # choose closer neighbor
                left = AI_TS_interp[idx - 1]
                right = AI_TS_interp[idx]

                idx -= (x - left) < (right - x)

                behTimeStamp['AlignedTimeStamp'] = AI_TS_aligned[idx]
                old_path = self.data_index['behTimeStamp'][ii]
                folder, old_file = os.path.split(old_path)
                new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                behTimeStamp.to_csv(new_file, index=False)
                self.data_index.loc[ii,'behTimeStamp'] = new_file


            if os.path.exists(self.data_index['ImgTimeStamp'][ii]):
                ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii], header=None)
                # define headers
                header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z']
                ImgTimeStamp.columns = header
                            
                # convert absolute time stamp (first column) to total minisecond, timeofday
                ts_temp = ImgTimeStamp['TimeStamp'].values
                x = [iso_to_timeofday(ts)*1000 for ts in ts_temp]
                idx = np.searchsorted(AI_TS_interp, x)

                # clip to valid range
                idx = np.clip(idx, 1, len(AI_TS_interp) - 1)

                # choose closer neighbor
                left = AI_TS_interp[idx - 1]
                right = AI_TS_interp[idx]

                idx -= (x - left) < (right - x)

                ImgTimeStamp['AlignedTimeStamp'] = AI_TS_aligned[idx]
                old_path = self.data_index['ImgTimeStamp'][ii]
                folder, old_file = os.path.split(old_path)
                new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                ImgTimeStamp.to_csv(new_file, index=False)
                self.data_index.loc[ii,'ImgTimeStamp'] = new_file

        # make plots for alignment checking
        # subplot 1: mismatch between AI_TS_interp and behTimeStamp, plus AI_TS_aligned
            plt.figure(figsize=(8,8))
            plt.subplot(2,2,1)
            x=behDF['outcome'][LC_Mask] - behDF['outcome'][LC_trialNum[0]]
            y=AI_TS_interp[matched]/1000-AI_TS_interp[matched[0]]/1000-x
            y_corrected = AI_TS_aligned[matched]-x- AI_TS_aligned[matched[0]]
            plt.plot(y)
            plt.plot(y_corrected)
            plt.title('Mismatch between Anolog Input and behavior')
            plt.xlabel('Trials')
            plt.ylabel('Time (s)')
            plt.legend(['Before correction', 'After correction'])

            savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii])
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)
            plt.savefig(os.path.join(savefigpath, 'TimeStamp_alignment.png'))
            plt.close()

    def cal_traces(self):
        # make some basic plot of calcium traces, aligned with behavior events
        # PSTH aligned to center_in, center_out (separate odor), side_in, outcome (separate choice/reward)
        # 
        nFiles = self.data_index.shape[0]
        
        for ii in range(nFiles):
            # load the behavior file and dFF file
            behDF = pd.read_csv(self.data_index['BehCSV'][ii])
            dFFFile = self.data_index['dFFFile'][ii]
            # load the aligned imaging time stamp
            ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii], header=None)
            header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z', 'AlignedTimeStamp']
            ImgTimeStamp.columns = header

            # load dFF file in .mat format
            dffResults = {}
            with h5py.File(dFFFile, "r") as f:
                A_grp = f['results']['A']
                #print(len(A_grp['jc']))
                #shape = A_grp['shape'][()]
                A = csc_matrix((
                    A_grp['data'][()],
                    A_grp['ir'][()],
                    A_grp['jc'][()]
                ), shape = (90000, len(A_grp['jc']) - 1))
                dffResults['A'] = A  # spatial contours of identified neurons

                dffResults['n'] = A.shape[1]
                dffResults['dFF'] = f['results']['C'][()]  # dF/F traces of identified neurons
             # save thses in pickels for later use
            savedffpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result')
            if not os.path.exists(savedffpath):
                os.makedirs(savedffpath)
                with open(os.path.join(savedffpath,'dFF_results.pkl'), 'wb') as f:
                    pickle.dump(dffResults, f)
             
            #%% plot PSTH for each neuron
            beh_events = ['center_in', 'center_out', 'side_in', 'outcome']
            trial_types_go = ['left', 'right'] 
            color_go = ['red', 'blue'] # if aligned to 'center_in' and 'center_out'
            trial_types_choice = ['left correct', 'right correct', 'left incorrect', 'right incorrect']
            color_choice = ['green', 'cyan', 'orange', 'purple'] # if aligned to 'side_in' and 'outcome'
            nCells = dffResults['n']
            nTrials = behDF.shape[0]
            t_start = -2
            t_end = 4
            interpT = np.arange(t_start, t_end, 0.05)
            
            dff_aligned = {}
            for event in beh_events:
                # align and interpolate dFF by the time of this event
                dff_aligned[event] = np.full((len(interpT), nTrials, nCells), np.nan)
                event_times = behDF[event].to_numpy()
                t_rel = interpT+event_times[:, np.newaxis]
                # ntrials x nTimePoints
                    # Interpolate each cell across all trials
                for trial_idx in range(nTrials):
                    # get the time of this event for this trial
                    event_start = t_rel[trial_idx, 0]
                    event_end = t_rel[trial_idx, -1]
                    # find the closest time in ImgTimeStamp to the event time
                    dff_timeStamp = np.array(ImgTimeStamp['AlignedTimeStamp'][1:].values, dtype=float)
                    timeMask = np.logical_and((dff_timeStamp >= event_start) ,
                        (dff_timeStamp <= event_end))
                    dff_temp =  dffResults['dFF'][timeMask, :]
                    t_ttemp = dff_timeStamp[timeMask]
                    # interpolate the dFF trace of this cell at the time points in interpT

                    if not np.sum(timeMask) == 0:
                        f = interp1d(
                                t_ttemp,
                                dff_temp,
                                axis=0,
                                bounds_error=False,
                                fill_value=np.nan
                            )

                        dff_aligned[event][:,trial_idx, :] = f(t_rel[trial_idx])


                #%% PSTH plot
                for cc in tqdm(range(nCells)):
                    plt.figure(figsize=(10,8))
                    # title for the whole figure
                    plt.suptitle(f'Neuron {cc}')
                    for i, event in enumerate(beh_events):
                        plt.subplot(2,2,i+1)
                        # make subplot
                        ### plotting PSTH for go/nogo/probe cues--------------------------------------------
                        if event in ['center_in', 'center_out']:
                            # look for trial_type_go
                            for tidx, trial in enumerate(trial_types_go):
                                trialMask = behDF['schedule']== (1 if trial=='left' else 2)
                                tempdFF = dff_aligned[event][:, trialMask, cc]
                                boot = bootstrap(tempdFF, 1, 1000)

                                plt.plot(interpT, boot['bootAve'], color=color_go[tidx], label=trial)
                                plt.fill_between(interpT, boot['bootLow'], boot['bootHigh'], color = color_go[tidx],label='_nolegend_', alpha=0.2)
                                
                                plt.title(event)

                        elif event in ['side_in', 'outcome']:
                            # look for trial_type_choice
                            for tidx, trial in enumerate(trial_types_choice):
                                trialMask = None
                                if trial == 'left correct':
                                    trialMask = np.logical_and(behDF['schedule']==1, behDF['reward']>0)
                                elif trial == 'right correct':
                                    trialMask = np.logical_and(behDF['schedule']==2, behDF['reward']>0)
                                elif trial == 'left incorrect':
                                    trialMask = np.logical_and(behDF['schedule']==1, np.isnan(behDF['reward']))
                                elif trial == 'right incorrect':
                                    trialMask = np.logical_and(behDF['schedule']==2, np.isnan(behDF['reward']))
                                tempdFF = dff_aligned[event][:, trialMask, cc]
                                boot = bootstrap(tempdFF, 1, 1000)

                                plt.plot(interpT, boot['bootAve'], color=color_choice[tidx], label=trial)
                                plt.fill_between(interpT, boot['bootLow'], boot['bootHigh'], color = color_choice[tidx],label='_nolegend_', alpha=0.2)
                            
                                plt.title(event)
                        ax = plt.gca()

                        # Remove top and right spines
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        # Remove legend box
                        ax.legend(frameon=False)
                        if i == 0 or i==2:
                            plt.ylabel('dF/F')
                        if i == 2 or i==3:
                            plt.xlabel('Time (s)')
                    savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Plot', 'PSTH')
                    os.makedirs(savefigpath, exist_ok=True)
                    plt.savefig(os.path.join(savefigpath, f'PSTH_Neuron_{cc}.png'), dpi=300, bbox_inches='tight')
                    plt.close()

            #%% what else?
if __name__ == "__main__":

    # use interactive matplotlib backend

    plt.ion()

    root_dir = r'Y:\HongliWang\Miniscope\ASD'

    Odor_Beh = BehDataOdor(root_dir)

    imaging_data = Imaging(root_dir, Odor_Beh)

    # basic plots of calcium traces aligned to behavior
    # 1. PSTH
    imaging_data.cal_traces()
    #%% analysis to do
    # 1. go over longitudinal registration
    # 2. align df/f with behavior events
