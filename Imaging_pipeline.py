from sklearn.linear_model import Ridge
import tqdm

from behavioral_pipeline import BehDataOdor, BehDataRotarod
import pandas as pd
import numpy as np
import os
import glob
import shutil

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from utils_imaging import *
from utils_rotarod import *
from scipy.sparse import csc_matrix
import h5py
import pickle
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score
from scipy.ndimage import center_of_mass

from tqdm import tqdm

from longitudinal import Longitudinal


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

        t0 = time.time()
        self.load_data()
        print(f"Data loading takes {time.time()-t0:.2f} seconds.")

        t0= time.time()
        self.align_timeStamps()
        print(f"Time stamp alignment takes {time.time()-t0:.2f} seconds.")
        
    def load_data(self):
        # load related imaging/recording data and timestamp file, according to behavior file
        nFiles = self.data_index.shape[0]

        if self.behavior == 'Odor':
            # go over Odor behavior
            for ii in range(nFiles):
                ImagingFolder = os.path.join(self.data,self.data_index['Animal'][ii],
                                            self.behavior,'Imaging', 
                                            self.data_index['Animal'][ii]+'_'+self.data_index['Date'][ii])


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
                if not dFFFile:
                    ifCalImg = False
                else:
                    ifCalImg = True
                
                # add these to the data_index
                #self.data_index.loc[ii,'ROIFile'] = ROIFile[0] if ROIFile else None
                self.data_index.loc[ii,'behRecording'] = behRecording[0] if behRecording else None
                self.data_index.loc[ii,'behTimeStamp'] = behTimeStamp[0] if behTimeStamp else None
                self.data_index.loc[ii,'AIMatrix'] = AIMatrix[0] if AIMatrix else None
                self.data_index.loc[ii,'AITimsStamp'] = AITimsStamp[0] if AITimsStamp else None
                self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp[0] if ImgTimeStamp else None
                self.data_index.loc[ii,'ifCalImg'] = ifCalImg
                self.data_index.loc[ii,'ifBehRecording'] = ifBehRecording
                self.data_index.loc[ii,'dFFFile'] = dFFFile[0] if dFFFile else None
                animal_index = np.where(self.Animals==self.data_index['Animal'][ii])[0][0]
                self.data_index.loc[ii, 'hemisphere'] = self.Hemisphere[animal_index]

        elif self.behavior == 'Rotarod':
                # go over Rotarod behavior
            for ii in range(nFiles):
                ImagingFolder = os.path.join(self.data,self.data_index['Animal'][ii],
                                            self.behavior,'Imaging', 
                                            self.data_index['Animal'][ii]+'_'+self.data_index['Date'][ii])

                # check if ROIFile exist

                
                # load behavior recordings
                behRecording = self.data_index['Video'][ii]

                # check if behRecording exist
                if not behRecording:
                    ifBehRecording = False
                else:
                    ifBehRecording = True


                behTimeStamp = self.data_index['BehTimestamp'][ii]
                    
                trial = self.data_index['Trial'][ii]
                ImgTimeStamp = glob.glob(os.path.join(ImagingFolder, f'*_ImgTimeStamp_trial{trial}*.csv'))
                dFFFile = glob.glob(os.path.join(ImagingFolder,'caiman_results', f'*_trial{trial}_updated_cnmf.mat'))

                if not dFFFile:
                    ifCalImg = False
                else:
                    ifCalImg = True
                # add these to the data_index
                self.data_index.loc[ii,'behRecording'] = behRecording[0] if behRecording else None
                #self.data_index.loc[ii,'behTimeStamp'] = behTimeStamp[0] if behTimeStamp else None
                self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp[0] if ImgTimeStamp else None
                self.data_index.loc[ii,'ifCalImg'] = ifCalImg
                self.data_index.loc[ii,'ifBehRecording'] = ifBehRecording
                self.data_index.loc[ii,'dFFFile'] = dFFFile[0] if dFFFile else None

    def align_timeStamps(self):
        # files to align
        # read TTL pulse, TTL timestamp, image timestamp, behTimeStamp and align them with behavior csv file
        # to do: rotarod time alignment

        nFiles = self.data_index.shape[0]

        if self.behavior == 'Odor':
            for ii in range(nFiles):
                # check if figure has been generated
                savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                    self.data_index['Date'][ii])
                savefigname = os.path.join(savefigpath, 'TimeStamp_alignment.png')
                if not os.path.exists(savefigname):
                    # if not exist, do the alignment
                    if not os.path.exists(savefigpath):
                        os.makedirs(savefigpath)
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
                    # check if it is aligned

                    if not os.path.exists(self.data_index['behTimeStamp'][ii]):
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
                        self.data_index.loc[ii,'behTimeStamp'] = new_file
                        behTimeStamp.to_csv(new_file, index=False)



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
                        self.data_index.loc[ii,'ImgTimeStamp'] = new_file
                        ImgTimeStamp.to_csv(new_file, index=False)
                        

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

                    plt.savefig(savefigname)
                    plt.close()
                    
                else:
                    # if already aligned, reset the timestamp files
                    # behavior
                    old_path = self.data_index['behTimeStamp'][ii]
                    folder, old_file = os.path.split(old_path)
                    new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                    self.data_index.loc[ii,'behTimeStamp'] = new_file 

                    # imaging
                    old_path = self.data_index['ImgTimeStamp'][ii]
                    folder, old_file = os.path.split(old_path)
                    new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                    self.data_index.loc[ii,'ImgTimeStamp'] = new_file

        elif self.behavior == 'Rotarod':
            # align rotorad timestamp (behavior, miniscope, and rod speed)
            for ii in range(nFiles):
                
                trial = self.data_index['Trial'][ii]
                savedatapath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging', f'trial{trial}',
                    'Processed_data')
                ImgTimeStamp_path = os.path.join(savedatapath, 'ImgTimeStamp_aligned.csv')
                behRecordingTimeStamp_path = os.path.join(savedatapath, 'behRecordingTimeStamp_aligned.csv')
                rodSpeed_smoothed_path = os.path.join(savedatapath, 'rodSpeed_smoothed.csv')
                # check if the aligned timestamp already exist
                if not os.path.exists(ImgTimeStamp_path):
                    if not os.path.exists(savedatapath):
                        os.makedirs(savedatapath)
                    # load the time stamps of behavior recording, rotarod speed, and calcium imaging
                    behRecordingTimeStamp = pd.read_csv(self.data_index['BehTimestamp'][ii], header=None)
                    rodSpeed = pd.read_csv(self.data_index['Rod_speed'][ii], header=None)
                    ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii], header=None)
                    header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z']
                    ImgTimeStamp.columns = header

                    savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging', f'trial{trial}')
                    label = f'{self.data_index["Animal"][ii]} trial{trial}'
                    # for rotarod alignment, all three time stamps are recorded on the same computer so is already aligned
                    # just convert iso time to timeofday in seconds
                    # set the time of rotarod start to 0
                    # so we need to process the rotarod speed data first

                    # smooth rod speed data
                    rodSpeed_smoothed = rodSpeed_smoothing(rodSpeed, label, savefigpath)
                    # save the rod speed
                    
                    pd.DataFrame(rodSpeed_smoothed).to_csv(rodSpeed_smoothed_path, index=False)
                    self.data_index.loc[ii,'Rod_speed'] = rodSpeed_smoothed_path

                    # align the rest to rod speed
                    time0 = rodSpeed_smoothed['time0'][0]
                    behRecordingTimeStamp['AlignedTimeStamp'] = behRecordingTimeStamp/1000 - time0
                    # save data
                    
                    behRecordingTimeStamp.to_csv(behRecordingTimeStamp_path, index=False)
                    self.data_index.loc[ii,'BehTimestamp'] = behRecordingTimeStamp_path

                    ts_temp = ImgTimeStamp['TimeStamp'].values
                    x = [iso_to_timeofday(ts) for ts in ts_temp]
                    ImgTimeStamp['AlignedTimeStamp'] = x - time0
                    # save data
                    
                    self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp_path
                    ImgTimeStamp.to_csv(ImgTimeStamp_path, index=False)
                else:
                    # if already aligned
                    self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp_path
                    self.data_index.loc[ii,'BehTimestamp'] = behRecordingTimeStamp_path
                    self.data_index.loc[ii,'Rod_speed'] = rodSpeed_smoothed_path
                
    
    def dff_to_pickle(self):
        # extract dFF traces from .mat and save it in pickels
        # also align dFF to behavior events center_in and side_in
        nFiles = self.data_index.shape[0]
            
        for ii in range(nFiles):
            # load the behavior file and dFF file
            dFFFile = self.data_index['dFFFile'][ii]

            savedffpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                    self.data_index['Date'][ii], 'Result')
            if not os.path.exists(savedffpath):
                os.makedirs(savedffpath)
            dff_pickle_path = os.path.join(savedffpath, 'dFF_results.pkl')
            if not os.path.exists(dff_pickle_path):
                         
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

                    with open(os.path.join(savedffpath,'dFF_results.pkl'), 'wb') as f:
                        pickle.dump(dffResults, f)

            self.data_index.loc[ii,'dFF_pickle'] = os.path.join(savedffpath,'dFF_results.pkl')

            #%% align dFF to center_in and side_in
            dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
            
            if not os.path.exists(dff_aligned_path):
                # load dffResults 
                with open(self.data_index['dFF_pickle'][ii], "rb") as f:
                    dffResults = pickle.load(f)
                
                behDF = pd.read_csv(self.data_index['BehCSV'][ii])
                    
                    # load the aligned imaging time stamp
                ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii], header=None)
                header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z', 'AlignedTimeStamp']
                ImgTimeStamp.columns = header

                beh_events = ['center_in', 'side_in']
                nCells = dffResults['n']
                nTrials = behDF.shape[0]
                t_start = -2
                t_end = 4
                interpT = np.arange(t_start, t_end, 0.05)
                t_start_long = t_start - 1.0  # 1s before
                t_end_long = t_end + 1.0      # 1s after
                interpT_long = np.arange(t_start_long, t_end_long, 0.05)

                dff_aligned = {}
                dff_aligned['time'] = interpT
                for event in beh_events:
                    # align and interpolate dFF by the time of this event
                    dff_aligned[event] = np.full((len(interpT), nTrials, nCells), np.nan)
                    event_times = behDF[event].to_numpy()
                    t_rel_long = interpT_long+event_times[:, np.newaxis]
                    # ntrials x nTimePoints
                        # Interpolate each cell across all trials
                    for trial_idx in range(nTrials):
                        # get the time of this event for this trial
                        event_time = event_times[trial_idx]
                        event_start = t_rel_long[trial_idx, 0]
                        event_end = t_rel_long[trial_idx, -1]
                        # find the closest time in ImgTimeStamp to the event time
                        dff_timeStamp = np.array(ImgTimeStamp['AlignedTimeStamp'][1:].values, dtype=float)
                        timeMask = np.logical_and((dff_timeStamp >= event_start) ,
                            (dff_timeStamp <= event_end))
                        dff_temp =  dffResults['dFF'][timeMask, :]
                        t_temp = dff_timeStamp[timeMask]
                        # interpolate the dFF trace of this cell at the time points in interpT

                        if t_temp.size >= 2:
                            f = interp1d(
                                    t_temp,
                                    dff_temp,
                                    axis=0,
                                    bounds_error=False,
                                    fill_value='extrapolate'
                                )

                            dff_aligned[event][:,trial_idx, :] =  f(interpT + event_time)

                # save dff_aligned

                with open(dff_aligned_path, "wb") as f:
                    pickle.dump(dff_aligned, f)

            self.data_index.loc[ii,'dff_aligned_pickle'] = dff_aligned_path

                        
    def cal_traces(self):

        # make some basic plot of calcium traces, aligned with behavior events
        # PSTH aligned to center_in, center_out (separate odor), side_in, outcome (separate choice/reward)
        # for rotarod, align to left/right stride

        #%% for odor behavior
        # in AB sessions - average the whole session
        # in AB-CD sessions - look for CD trials only
        if self.behavior == 'Odor':
            nFiles = self.data_index.shape[0]
            
            for ii in range(nFiles):
                # load the behavior file and dFF file
                behDF = pd.read_csv(self.data_index['BehCSV'][ii])
                protocol = self.data_index['Protocol'][ii]

                #%% plot PSTH for each neuron
                beh_events = ['center_in','side_in']
                trial_types_go = ['A contra', 'A ipsi', 'B contra', 'B ipsi'] 
                color_go = ['red', 'blue'] # if aligned to 'center_in' and 'center_out'
                trial_types_choice = ['contra correct', 'ipsi correct', 'contra incorrect', 'ipsi incorrect']
                color_choice = ['green', 'cyan', 'orange', 'purple'] # if aligned to 'side_in' and 'outcome'

                dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
                with open(dff_aligned_path, "rb") as f:
                    dff_aligned = pickle.load(f)
                nCells = dff_aligned['center_in'].shape[2]
                interpT = dff_aligned['time']
                nTrials = dff_aligned['center_in'].shape[1]

                # determine contra and ipsi
                if self.data_index['hemisphere'][ii] == 'Right':
                    contra_action = 1
                    ipsi_action = 0
                else:
                    contra_action = 0
                    ipsi_action = 1

                # based on schedule, look for the right trials
                if protocol == 'AB':
                    schedule_A = 1
                    schedule_B = 2
                elif protocol == 'AB-CD':
                    schedule_A = 3
                    schedule_B = 4  # look for C, D trials

                #%% PSTH plot
 
                for cc in tqdm(range(nCells)):
                    plt.figure(figsize=(10,8))
                    # title for the whole figure
                    plt.suptitle(f'Neuron {cc}')
                    
                    plt.subplot(2,1,1)

                    ### plotting PSTH for A/B odor cues
            
                    for tidx, trial in enumerate(trial_types_go):
                        trialMask = np.logical_and(behDF['schedule']== (schedule_A if 'A' in trial else schedule_B),
                                                    behDF['actions'] == (contra_action if 'contra' in trial else ipsi_action))
                        
                        tempdFF = dff_aligned['center_in'][:, trialMask, cc]
                        boot = bootstrap(tempdFF, 1, 1000)

                        plt.plot(interpT, boot['bootAve'], color=color_choice[tidx], label=trial)
                        plt.fill_between(interpT, boot['bootLow'], boot['bootHigh'], color = color_choice[tidx],label='_nolegend_', alpha=0.2)
                        
                        plt.title('Stimulus aligned to center in')
                        
                    ax = plt.gca()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    # Remove legend box
                    ax.legend(frameon=False)
                    plt.ylabel('dF/F')
                    

                    plt.subplot(2,1,2)
                    # outcome aligned to side in
                    for tidx, trial in enumerate(trial_types_choice):
                        trialMask = None
                        if trial == 'contra correct':
                            trialMask = ((behDF['actions']==contra_action) &
                                           (behDF['reward']>0) &
                                            (behDF['schedule'].isin([schedule_A, schedule_B])))
                        elif trial == 'ipsi correct':
                            trialMask = ((behDF['actions']==ipsi_action) &
                                           (behDF['reward']>0) &
                                            (behDF['schedule'].isin([schedule_A, schedule_B])))
                        elif trial == 'contra incorrect':
                            trialMask = ((behDF['actions']==contra_action) &
                                           (np.isnan(behDF['reward'])) &
                                            (behDF['schedule'].isin([schedule_A, schedule_B])))
                        elif trial == 'ipsi incorrect':
                            trialMask = ((behDF['actions']==ipsi_action) &
                                           (np.isnan(behDF['reward'])) &
                                            (behDF['schedule'].isin([schedule_A, schedule_B])))
                        tempdFF = dff_aligned['side_in'][:, trialMask, cc]
                        boot = bootstrap(tempdFF, 1, 1000)

                        plt.plot(interpT, boot['bootAve'], color=color_choice[tidx], label=trial)
                        plt.fill_between(interpT, boot['bootLow'], boot['bootHigh'], color = color_choice[tidx],label='_nolegend_', alpha=0.2)
                    
                        plt.title('outcome aligned to side_in')
                        ax = plt.gca()

                        # Remove top and right spines
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        # Remove legend box
                        ax.legend(frameon=False)
                        plt.xlabel('Time (s)')
                        plt.ylabel('dF/F')

                    savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Plot', 'PSTH')
                    os.makedirs(savefigpath, exist_ok=True)
                    plt.savefig(os.path.join(savefigpath, f'PSTH_Neuron_{cc}.png'), dpi=300, bbox_inches='tight')
                    plt.close()

    def auROC(self):
        # long_data: longitudinal coregistration 

        # calculate auROC for each neuron to determine the selectivity 
        # for stimulus/choice/reward
        # stimulus: 0-1s, aligned to center_in, also seprate into contra/ipsi trials
        # choice: 0-4s, aligned to center_in
        # reward: 0-2s, aligned to side_in
        nFiles = self.data_index.shape[0]

        # temp varible for plot
        n_sig_AB = np.full((1, nFiles), np.nan)
        n_sig_CD = np.full((1, nFiles), np.nan)
        perf_AB = np.full((1, nFiles), np.nan)
        perf_CD = np.full((1, nFiles), np.nan)
        d_AB = np.full((1, nFiles), np.nan)
        d_CD = np.full((1, nFiles), np.nan)
        crit_AB = np.full((1, nFiles), np.nan)
        crit_CD = np.full((1, nFiles), np.nan) # criterion for d prime (bias)

        for ii in range(nFiles):
            dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
            with open(dff_aligned_path, "rb") as f:
                dff_aligned = pickle.load(f)
            savedatapath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                    self.data_index['Date'][ii], 'Result')
            behDF = pd.read_csv(self.data_index['BehCSV'][ii])
            perf_AB[0,ii] = np.nanmean(behDF['reward'][(behDF['schedule']==1) | (behDF['schedule']==2)]>0)
            perf_CD[0,ii] = np.nanmean(behDF['reward'][(behDF['schedule']==3) | (behDF['schedule']==4)]>0)
            
            # calculate d_prime
            d_data_AB = {}
            d_data_AB['stimulus'] = np.array(behDF['schedule'][(behDF['schedule']==1) | (behDF['schedule']==2)]).astype(int)
            d_data_AB['actions'] = np.array(behDF['actions'][(behDF['schedule']==1) | (behDF['schedule']==2)]).astype(int)+1
            [d_AB[0,ii], crit_AB[0,ii]] = d_prime(d_data_AB)
            if self.data_index['Protocol'][ii] == 'AB-CD':
                d_data_CD = {}
                d_data_CD['stimulus'] = np.array(behDF['schedule'][(behDF['schedule']==3) | (behDF['schedule']==4)]).astype(int)-2
                d_data_CD['actions'] = np.array(behDF['actions'][(behDF['schedule']==3) | (behDF['schedule']==4)]).astype(int)+1
                [d_CD[0,ii], crit_CD[0,ii]] = d_prime(d_data_CD)

            savedataname = os.path.join(savedatapath, 
                                        f"auROC_results_stimulus_{self.data_index['Protocol'][ii]}.pkl")
            # determine contra/ipsi side
            if os.path.exists(savedataname):
                if self.data_index['hemisphere'][ii] == 'Right':
                    contra_action = 0
                    ipsi_action = 1
                else:
                    contra_action = 1
                    ipsi_action = 0
                # calculate auROC for each cell 
                # for stimulus/choice/outcome
                if self.data_index['Protocol'][ii] == 'AB':
                    schedule_A = 1
                    schedule_B = 2
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    schedule_A = 3
                    schedule_B = 4  # look for C, D trials
                #%% auROC calculation for stimulus
                # 0-1 s only
                dff_included = dff_aligned['center_in'][(dff_aligned['time']<2) & (dff_aligned['time']>0), :,:]
                nTime, nTrials, nCells = dff_included .shape

                # supT test to determine neurons that encode stimulus (aligned to center_in)
                # for contral trials
                #dff_contra = np.nanmean(dff_included[:, trials_contra, :],0, keepdims=True)
                # include the trials with targeted odor only (exclude AB trials in AB-CD) sessions for now
                trial_include_mask = np.logical_or(behDF['schedule']==schedule_A, behDF['schedule']==schedule_B)
                labels = {}
                # for auROC, we need the choice label to stratify trials
                # calculate auROC for stimulus, but use choice for stratify
                labels['target'] = np.array(behDF['schedule'][trial_include_mask]==schedule_A).astype(int)
                labels['stratify'] = np.array(behDF['actions'][trial_include_mask]==1).astype(int)  # B = 1, else = 0
                dff_included=dff_included[:, trial_include_mask, :]
                [auc_s, p_auc_s_adjusted, auc_s_null] = supT_stats(dff_included, labels, auROC_supT, 1000)
                
                # for AB-CD sessions, also look at AB trials
                if self.data_index['Protocol'][ii] == 'AB-CD':
                    # include the AB trials
                    trial_include_mask_AB = np.logical_or(behDF['schedule']==1, behDF['schedule']==2)
                    dff_tmp = dff_aligned['center_in'][(dff_aligned['time']<2) & (dff_aligned['time']>0), :, :]
                    dff_included_AB = dff_tmp[:, trial_include_mask_AB, :]
                    labels_AB = {}
                    labels_AB['target'] = np.array(behDF['schedule'][trial_include_mask_AB]==1).astype(int)
                    labels_AB['stratify'] = np.array(behDF['actions'][trial_include_mask_AB]==1).astype(int)  # B = 1, else = 0
                    [auc_s_AB, p_auc_s_adjusted_AB, auc_s_null_AB] = supT_stats(dff_included_AB, labels_AB, auROC_supT, 1000)
                # save the file
                if self.data_index['Protocol'][ii] == 'AB':
                    n_sig_AB[0, ii] = np.sum(p_auc_s_adjusted<0.05)/ nCells
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    n_sig_AB[0, ii] = np.sum(p_auc_s_adjusted_AB<0.05)/nCells
                    n_sig_CD[0, ii] = np.sum(p_auc_s_adjusted<0.05)/nCells

                auROC_results = {
                    'auc_s': auc_s,
                    'p_auc_s_adjusted': p_auc_s_adjusted,
                    'auc_s_null': auc_s_null,
                    'auc_s_AB': auc_s_AB if self.data_index['Protocol'][ii] == 'AB-CD' else None,
                    'p_auc_s_adjusted_AB': p_auc_s_adjusted_AB if self.data_index['Protocol'][ii] == 'AB-CD' else None,
                    'auc_s_null_AB': auc_s_null_AB if self.data_index['Protocol'][ii] == 'AB-CD' else None
                }
                with open(savedataname, 'wb') as f:
                    pickle.dump(auROC_results, f)

            else:
                # load results
                with open(savedataname, 'rb') as f:
                    auROC_results = pickle.load(f)
                nCells = dff_aligned['center_in'].shape[2]
                if self.data_index['Protocol'][ii] == 'AB':
                    n_sig_AB[0, ii] = np.sum(auROC_results['p_auc_s_adjusted']<0.05)/ nCells
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    n_sig_AB[0, ii] = np.sum(auROC_results['p_auc_s_adjusted_AB']<0.05)/nCells
                    n_sig_CD[0, ii] = np.sum(auROC_results['p_auc_s_adjusted']<0.05)/nCells


            # make a plot temporarily 
        fig, ax1 = plt.subplots()
        plt.title('AUROC (stimulus) selectivity in 0-2s after center in')
        AB_plot = np.arange(0,4,1)
        CD_plot = np.arange(4,7,1)
        ax1.plot(AB_plot, n_sig_AB[0,AB_plot], color='red')
        ax1.plot(CD_plot, n_sig_CD[0,CD_plot], color = 'blue')
        ax1.set_ylabel('Fraction of neurons selective for stimulus')

        ax2 = ax1.twinx()
        ax2.plot(AB_plot, d_AB[0,AB_plot], '--', color='red')
        ax2.plot(CD_plot, d_CD[0,CD_plot], '--',color='blue')
        ax2.set_ylabel('d prime for behavior performance')

        x_labels = ['AB', 'AB', 'AB', 'AB', 'CD', 'CD', 'CD']

        x_positions = np.arange(7)

        # Labels for each tick
        x_labels = ['AB', 'AB', 'AB', 'AB', 'CD', 'CD', 'CD']

        # Set ticks on the primary axis
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_labels)
        
        #%% look for coregistered neurons acrss sessions if long_data is provided
        if long_data is not None:
            pass

    def MLR_session(self):
        # generalized linear regression for 
        # stimulus/choice/latent variables
        nFiles = self.data_index.shape[0]
        for ii in range(nFiles):
            savefigfolder = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Plot')
            savedatafolder = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result')
            savedatapath = os.path.join(savedatafolder, 'MLR_results.pkl')
            if not os.path.exists(savedatapath):
                dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                    self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
                with open(dff_aligned_path, "rb") as f:
                    dff_aligned = pickle.load(f)
                nTrials = dff_aligned['center_in'].shape[1]
                nCells = dff_aligned['center_in'].shape[2]
                # average within 0.1 s window
                regr_time = np.arange(-1.95, 3.95 + 1e-9, 0.1)
                dff_regr = np.full((len(regr_time), nTrials, nCells), np.nan)
                for tIdx, t in enumerate(regr_time):
                    tstart = t-0.1/2
                    tend = t+0.1/2
                    dff_regr[tIdx, :,:] = np.nanmean(dff_aligned['center_in'][np.logical_and(dff_aligned['time']<=tend, dff_aligned['time']>=tstart), :,:], axis=0)

                # find trials to include based on schedule
                
                behDF = pd.read_csv(self.data_index['BehCSV'][ii])
                if self.data_index['Protocol'][ii] == 'AB':
                    schedule_A = 1
                    schedule_B = 2
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    schedule_A = 3
                    schedule_B = 4  # look for C, D trials

                trial_include_mask = np.logical_or(behDF['schedule']==schedule_A, behDF['schedule']==schedule_B)
                nTrials = np.sum(trial_include_mask)
                dff_regr = dff_regr[:, trial_include_mask, :]
                # determine contra/ipsi side
                if self.data_index['hemisphere'][ii] == 'Right':
                    contra_action = 0
                    ipsi_action = 1
                else:
                    contra_action = 1
                    ipsi_action = 0
                
                # prepare the predictor matrix 
                # for now: Sn + Cn + SnxCn + Sn-1 + Cn-1 + Sn-1xCn-1
                #pred_names = ['sn', 'cn', 'sn*cn', 'sn-1', 'cn-1', 'sn-1*cn-1']
                pred_names = ['sn', 'cn', 'sn*cn', 'sn-1', 'cn-1', 'sn-1*cn-1']
                nPred = len(pred_names)
                pred_mat = np.full((nTrials, nPred), np.nan)
                s = (behDF['schedule'][trial_include_mask]-1.5)*2 # A: -1; B: 1
                if contra_action==1:
                    c = behDF['actions'][trial_include_mask] 
                    c[behDF['actions'][trial_include_mask]==0] = -1
                else:
                    c = behDF['actions'][trial_include_mask] 
                    c[behDF['actions'][trial_include_mask]==0] = 1 # contra 1, ipsi -1
                    c[behDF['actions'][trial_include_mask]==1] = -1
                for pIdx, pred in enumerate(pred_names):
                    if pred == 'sn':
                        pred_mat[:,pIdx] = s
                    elif pred == 'cn':
                        pred_mat[:,pIdx] = c
                    elif pred == 'sn*cn':
                        pred_mat[:,pIdx] = s*c
                    elif pred == 'sn-1':
                        pred_mat[:,pIdx] = np.concatenate(([np.nan], s[0:-1]))
                    elif pred == 'cn-1':
                        pred_mat[:,pIdx] = np.concatenate(([np.nan], c[0:-1]))
                    elif pred == 'sn-1*cn-1':
                        pred_mat[:,pIdx] = np.concatenate(([np.nan], s[0:-1]*c[0:-1])) 
                                
                [linear_results, p_adjusted, linear_null] = supT_stats(dff_regr, pred_mat, linear_regr_supT, 1000)

                #plot the regression results, save the neuron identify of signifacant neurons of each variable
                pred = ['stimulus n', 'choice n', 'outcome n', 'stimulus n-1', 'choice n-1', 'outcome n-1']
                n_pred = len(pred_names)
                fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
                axes = axes.flatten()  
                for p in range(n_pred):
                    ax = axes[p]
                    coeff = np.sum(linear_results['pvalue'][:, :, p+1]<0.01,1)/linear_results['pvalue'].shape[1] # skip intercept
                    ax.plot(regr_time, coeff)
                    ax.set_title(pred[p])
                    # remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    if p>=3:
                        ax.set_xlabel('Time')
                    if p==0 or p==3:
                        ax.set_ylabel('Percentage signifcant cells ')
                
                # save the results
                with open(savedatapath, 'wb') as f:
                    pickle.dump({'linear_results': linear_results, 'p_adjusted': p_adjusted, 'linear_null': linear_null}, f)
                
                # save the figure in png and vector format
                plt.savefig(os.path.join(savefigfolder, 'MLR_results.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(savefigfolder, 'MLR_results.svg'), bbox_inches='tight')

    def MLR_orthogonal(self):
        # orthogonalize the predictors and disentangle stimulus/choice/outcome
        nFiles = self.data_index.shape[0]
        for ii in range(nFiles):
            ii=3
            dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
            with open(dff_aligned_path, "rb") as f:
                dff_aligned = pickle.load(f)
            nTrials = dff_aligned['center_in'].shape[1]
            nCells = dff_aligned['center_in'].shape[2]
            # average within 0.1 s window
            regr_time = np.arange(-1.95, 3.95 + 1e-9, 0.1)
            dff_regr = np.full((len(regr_time), nTrials, nCells), np.nan)
            for tIdx, t in enumerate(regr_time):
                tstart = t-0.1/2
                tend = t+0.1/2
                dff_regr[tIdx, :,:] = np.nanmean(dff_aligned['center_in'][np.logical_and(dff_aligned['time']<=tend, dff_aligned['time']>=tstart), :,:], axis=0)

            
            behDF = pd.read_csv(self.data_index['BehCSV'][ii])

            # determine contra/ipsi side
            if self.data_index['hemisphere'][ii] == 'Right':
                contra_action = 0
                ipsi_action = 1
            else:
                contra_action = 1
                ipsi_action = 0
            
            # prepare the predictor matrix 
            # for now: Sn + Cn + SnxCn + Sn-1 + Cn-1 + Sn-1xCn-1
            pred_names = ['sn', 'cn', 'sn*cn', 'sn-1', 'cn-1', 'sn-1*cn-1']
            nPred = len(pred_names)
            pred_mat = np.full((nTrials, nPred), np.nan)
            s = np.array((behDF['schedule']-1.5)*2) # A: -1; B: 1
            if contra_action==1:
                c = behDF['actions'] 
                c[behDF['actions']==0] = -1
            else:
                c = behDF['actions'] 
                c[behDF['actions']==0] = 1 # contra 1, ipsi -1
                c[behDF['actions']==1] = -1
            c = np.array(c)
            # construct the predictor matrix
            # remove linear dependence of c on s
            o = s*c
            valid_idx = ~np.isnan(s) & ~np.isnan(c)  # only valid trials
            s_valid = s[valid_idx]
            c_valid = c[valid_idx]
            # --- 2. orthogonalize s relative to c ---
            # Solve least squares: c -> s (remove part of s predictable by c)
            b_s, _, _, _ = np.linalg.lstsq(c_valid[:, np.newaxis], s_valid, rcond=None)
            s_orth = np.full_like(s, np.nan)
            s_orth[valid_idx] = s_valid - c_valid * b_s[0]

            # --- 3. orthogonalize c relative to s_orth ---
            b_c, _, _, _ = np.linalg.lstsq(s_orth[valid_idx][:, np.newaxis], c_valid, rcond=None)
            c_orth = np.full_like(c, np.nan)
            c_orth[valid_idx] = c_valid - s_orth[valid_idx] * b_c[0]

            pred_mat[:, 0] = s_orth
            pred_mat[:, 1] = c_orth
            pred_mat[:, 2] = o
            pred_mat[1:, 3] = s_orth[0:-1]
            pred_mat[1:, 4] = c_orth[0:-1]
            pred_mat[1:, 5] = o[0:-1]
            pred_mat[0, 3:] = np.nan  # first trial has no previous

            [linear_results, p_adjusted, linear_null] = supT_stats(dff_regr, pred_mat, linear_regr_supT, 1000)

    def GLM_time_kernel(self):
        # estimate the time resolved kernel
        # for stimulus/choice/outcome
        nFiles = self.data_index.shape[0]
        for ii in range(nFiles):
            # try to use the matlab gl_regr
            raw_dFF_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result', 'dFF_results.pkl')
            with open(raw_dFF_path, "rb") as f:
                dffResults = pickle.load(f)
            ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii], header=None)
            header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z', 'AlignedTimeStamp']
            ImgTimeStamp.columns = header

            behDF = pd.read_csv(self.data_index['BehCSV'][ii])

            dFF_signal = dffResults['dFF']
            dff_time = np.array(ImgTimeStamp['AlignedTimeStamp'][1:].values, dtype=float)
            trig_time = np.column_stack([behDF['center_in'].values-1, behDF['center_in'].values+2])


            # determine contra/ipsi side
            if self.data_index['hemisphere'][ii] == 'Right':
                contra_action = 0
                ipsi_action = 1
            else:
                contra_action = 1
                ipsi_action = 0
            
            event_names = ['stimulus', 'choice', 'outcome']
            event_times = {}
            event_times['stimulus'] = behDF['center_in'].values
            event_times['choice'] = behDF['center_in'].values
            event_times['outcome'] = behDF['side_in'].values

            params = {}
            params['window'] = [-1, 2]  # seconds relative to event for kernel
            params['eventLabel'] = event_names

            s = np.array((behDF['schedule']-1.5)*2) # A: -1; B: 1
            if contra_action==1:
                c = behDF['actions'] 
                c[behDF['actions']==0] = -1
            else:
                c = behDF['actions'] 
                c[behDF['actions']==0] = 1 # contra 1, ipsi -1
                c[behDF['actions']==1] = -1
            c = np.array(c)
            # construct the predictor matrix
            # remove linear dependence of c on s
            o = s*c

            from sklearn.linear_model import Ridge

            # ---------------------------
            # --- Set up variables ------
            # ---------------------------
            # dFF_signal: frames x nCells
            # dfF_time: frame times
            # trig_time: nTrials x 2 (start/end times for each trial)
            # event_times: dictionary with event arrays
            # s, c, o: event values per trial
            # params: window and labels

            # Parameters
            frame_rate = 1/np.median(np.diff(dff_time))  # Hz
            window_start, window_end = params['window']
            kernel_len = int(round((window_end - window_start) * frame_rate)) + 1  # number of time points in kernel
            time_lags = np.arange(window_start, window_end, 1/frame_rate)  # for kernel

            # Prepare basis for each event
            basisEvent = {}
            for name in event_names:
                vals = event_times[name]
                # Map event values to frame times
                b = np.zeros(len(dff_time))
                for trial_idx, t in enumerate(vals):
                    frame_idx = np.argmin(np.abs(dff_time - t))
                    if name == 'stimulus':
                        b[frame_idx] = s[trial_idx]  # assign value for stimulus
                    elif name == 'choice':
                        b[frame_idx] = c[trial_idx]
                    elif name == 'outcome':
                        b[frame_idx] = o[trial_idx]
                basisEvent[name] = b

            # ---------------------------
            # --- Construct full design matrix X ------
            # ---------------------------
            X_list = []
            mask = {}

            for k, name in enumerate(event_names):
                predictor = np.zeros((len(dff_time), kernel_len))  # initialize with 0
                
                for j, lag in enumerate(time_lags):
                    lag_frames = int(round(lag * frame_rate))
                    
                    if lag_frames < 0:  # neural activity precedes event
                        if -lag_frames < len(basisEvent[name]):
                            predictor[-lag_frames:, j] = basisEvent[name][:len(basisEvent[name])+lag_frames]
                        else:
                            predictor[:, j] = 0  # entire column is beyond bounds
                    elif lag_frames >= 0:  # neural activity follows event
                        if lag_frames < len(basisEvent[name]):
                            predictor[:len(basisEvent[name])-lag_frames, j] = basisEvent[name][lag_frames:]
                        else:
                            predictor[:, j] = 0  # entire column is beyond bounds
                
                X_list.append(predictor)
                
                # mask for this event
                mask[name] = np.zeros(len(event_names) * kernel_len, dtype=bool)
                mask[name][k*kernel_len:(k+1)*kernel_len] = True

            # Combine predictors
            X = np.hstack(X_list)

            # ---------------------------
            # --- Train/test split ------
            # ---------------------------
            nTrials = trig_time.shape[0]
            num_test = int(round(0.2*nTrials))
            drawNum = np.random.permutation(nTrials)
            testTrials = drawNum[:num_test]
            trainTrials = drawNum[num_test:]

            # Map trials to frame indices
            trainIdx = np.zeros(len(dff_time), dtype=bool)
            for tr in trainTrials:
                start_idx = np.argmin(np.abs(dff_time - trig_time[tr, 0]))
                end_idx = np.argmin(np.abs(dff_time - trig_time[tr, 1]))
                trainIdx[start_idx:end_idx+1] = True

            testIdx = np.zeros(len(dff_time), dtype=bool)
            for tr in testTrials:
                start_idx = np.argmin(np.abs(dff_time - trig_time[tr, 0]))
                end_idx = np.argmin(np.abs(dff_time - trig_time[tr, 1]))
                testIdx[start_idx:end_idx+1] = True

            # remove NaNs
            # y is dFF for one cell
            y_cell = dFF_signal[:, cell]  # shape: (time,)

            valid_idx = ~np.isnan(y_cell) & ~np.isnan(X).any(axis=1)

            X_valid = X[valid_idx, :]
            y_valid = y_cell[valid_idx]
            train_valid = trainIdx & ~np.isnan(y_cell) & ~np.isnan(X).any(axis=1)
            test_valid = testIdx & ~np.isnan(y_cell) & ~np.isnan(X).any(axis=1) 
            # ---------------------------
            # --- Ridge regression ------
            # ---------------------------
            # Choose lambda via cross-validation
            testLambda = 10.0 ** np.arange(-5, 0, 0.5)  # e.g., 1e-5 to 1
            best_lambda = testLambda[0]
            best_score = -np.inf

            nCells = dFF_signal.shape[1]

            for lam in testLambda:
                scores = []
                for cell in range(nCells):
                    y_cell = dFF_signal[:, cell]
                    train_valid = trainIdx & ~np.isnan(y_cell) & ~np.isnan(X).any(axis=1)
                    test_valid = testIdx & ~np.isnan(y_cell) & ~np.isnan(X).any(axis=1)
                    
                    if np.sum(train_valid) == 0 or np.sum(test_valid) == 0:
                        continue
                    
                    model = Ridge(alpha=lam, fit_intercept=True)
                    model.fit(X[train_valid, :], y_cell[train_valid])
                    y_pred = model.predict(X[test_valid, :])
                    scores.append(np.corrcoef(y_cell[test_valid], y_pred)[0, 1])
                
                mean_score = np.nanmean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_lambda = lam

            print("Best lambda:", best_lambda)
            print("Best mean CC:", best_score)

            # Fit final model with best lambda
            model = Ridge(alpha=best_lambda, fit_intercept=True)
            model.fit(X[train_valid, :], dFF_signal[train_valid, :])

            # Predicted signal for test set
            y_test_fit = np.full_like(dFF_signal, np.nan)
            y_test_fit[testIdx, :] = model.predict(X[testIdx, :])

            # Extract beta kernels
            nCells = dFF_signal.shape[1]
            kernels = np.full((kernel_len, nCells, len(event_names)), np.nan)
            for i, name in enumerate(event_names):
                kernels[:, :, i] = model.coef_[:, mask[name]].T  # reshape to kernel_len x nCells

            # ---------------------------
            # --- Output ------
            # ---------------------------
            # kernels: kernel_len x nCells x nEvents
            # y_test_fit: predicted signal on test frames
            # best_lambda: regularization parameter used


    def plotMLRResult(self, MLRResult, labels, neuroRaw, saveFigPath):
        # get the average coefficient plot and fraction of significant neurons
        varList =labels
        # average coefficient
        nPredictors = MLRResult['coeff'].shape[0]

        coeffPlot = StartSubplots(4,4, ifSharey=True)

        maxY = 0
        minY = 0
        for n in range(nPredictors):
            tempBoot = bootstrap(MLRResult['coeff'][n,:,:],1, 1000)
            tempMax = max(tempBoot['bootHigh'])
            tempMin = min(tempBoot['bootLow'])
            if tempMax > maxY:
                maxY = tempMax
            if tempMin < minY:
                minY = tempMin
            coeffPlot.ax[n//4, n%4].plot(MLRResult['regr_time'], tempBoot['bootAve'], c =(0,0,0))
            coeffPlot.ax[n // 4, n % 4].fill_between(MLRResult['regr_time'], tempBoot['bootLow'], tempBoot['bootHigh'],
                                          alpha=0.2,  color = (0.7,0.7,0.7))
            coeffPlot.ax[n//4, n%4].set_title(varList[n])
        plt.ylim((minY,maxY))
        plt.show()
        coeffPlot.save_plot('Average coefficient.tif','tiff', saveFigPath)

        # fraction of significant neurons
        sigPlot = StartSubplots(4, 4, ifSharey=True)
        pThresh = 0.001
        nCell = MLRResult['coeff'].shape[2]

        # binomial test to determine signficance

        for n in range(nPredictors):
            fracSig = np.sum(MLRResult['pval'][n, :, :]<pThresh,1)/nCell
            pResults = [binomtest(x,nCell,p=pThresh).pvalue for x in np.sum(MLRResult['pval'][n, :, :]<pThresh,1)]
            sigPlot.ax[n // 4, n % 4].plot(MLRResult['regr_time'], fracSig, c=(0, 0, 0))
            sigPlot.ax[n // 4, n % 4].set_title(varList[n])

            if n//4 == 0:
                sigPlot.ax[n // 4, n % 4].set_ylabel('Fraction of sig')
            if n > 8:
                sigPlot.ax[n // 4, n % 4].set_xlabel('Time from cue (s)')
            # plot the signifcance bar
            dt = np.mean(np.diff(MLRResult['regr_time']))
            for tt in range(len(MLRResult['regr_time'])):
                if pResults[tt]<0.05:
                    sigPlot.ax[n//4, n%4].plot(MLRResult['regr_time'][tt]+dt*np.array([-0.5,0.5]), [0.5,0.5],color=(255/255, 189/255, 53/255), linewidth = 5)
        plt.ylim((0,0.6))
        plt.show()
        sigPlot.save_plot('Fraction of significant neurons.tif', 'tiff', saveFigPath)

        # plot r-square
        r2Boot = bootstrap(MLRResult['r2'], 1, 1000)
        r2Plot = StartPlots()
        r2Plot.ax.plot(MLRResult['regr_time'], r2Boot['bootAve'],c=(0, 0, 0))
        r2Plot.ax.fill_between(MLRResult['regr_time'], r2Boot['bootLow'], r2Boot['bootHigh'],
                                                 color=(0.7, 0.7, 0.7))
        r2Plot.ax.set_title('R-square')
        r2Plot.save_plot('R-square.tif', 'tiff', saveFigPath)


        """plot significant neurons"""
        sigCells = MLRResult['sigCells']
        cellstat = []
        for cell in range(neuronRaw.Fraw.shape[0]):
            if neuronRaw.cells[cell, 0] > 0:
                cellstat.append(neuronRaw.stat[cell])

        fluoCellPlot = StartPlots()
        im = np.zeros((256, 256,3))

        #for cell in range(decode_results[var]['importance'].shape[0]):
        for cell in range(len(cellstat)):
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
            if cell not in \
                    set(sigCells['choice'])|set(sigCells['outcome'])|set(sigCells['stimulus']):
                im[ys, xs] = [0.7, 0.7, 0.7]

        for cell in sigCells['choice']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys,xs] = [0,0,0]
            im[ys, xs] = np.add(im[ys, xs], [1.0, 0.0, 0.0])
        for cell in sigCells['outcome']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,1.0,0.0])
        for cell in sigCells['stimulus']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,0.0,1.0])
        action_patch = mpatches.Patch(color=(1,0,0), label='Action')
        outcome_patch = mpatches.Patch(color=(0,1,0), label = 'Outcome')
        stimulus_patch = mpatches.Patch(color=(0, 0, 1), label='Stimulus')
        # Create a custom legend with the green patch
        plt.legend(handles=[action_patch, outcome_patch, stimulus_patch],loc='center left',bbox_to_anchor=(1, 0.5))
        fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        plt.show()

        fluoCellPlot.save_plot('Regression neuron coordinates.tiff', 'tiff', saveFigPath)



    def GLM_summary(self):
        # summarize the GLM result for each session
        pass

    def decoding_session(self):
        # decode task-relavant variables for each session
        nFiles = self.data_index.shape[0]
        for ii in range(nFiles):
            savefigfolder = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Plot')
            savedatafolder = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                self.data_index['Date'][ii], 'Result')
            savedatapath = os.path.join(savedatafolder, 'decoding_results.pkl')
            if not os.path.exists(savedatapath):
                dff_aligned_path = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                    self.data_index['Date'][ii], 'Result', 'dff_aligned.pkl')
                with open(dff_aligned_path, "rb") as f:
                    dff_aligned = pickle.load(f)
                nTrials = dff_aligned['center_in'].shape[1]
                nCells = dff_aligned['center_in'].shape[2]
                # average within 0.1 s window
                regr_time = np.arange(-1.95, 3.95 + 1e-9, 0.1)
                nTime = len(regr_time)
                dff_regr = np.full((len(regr_time), nTrials, nCells), np.nan)
                for tIdx, t in enumerate(regr_time):
                    tstart = t-0.1/2
                    tend = t+0.1/2
                    dff_regr[tIdx, :,:] = np.nanmean(dff_aligned['center_in'][np.logical_and(dff_aligned['time']<=tend, dff_aligned['time']>=tstart), :,:], axis=0)

                # find trials to include based on schedule
                
                behDF = pd.read_csv(self.data_index['BehCSV'][ii])
                if self.data_index['Protocol'][ii] == 'AB':
                    schedule_A = 1
                    schedule_B = 2
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    schedule_A = 3
                    schedule_B = 4  # look for C, D trials

                trial_include_mask = np.logical_or(behDF['schedule']==schedule_A, behDF['schedule']==schedule_B)
                nTrials = np.sum(trial_include_mask)
                dff_regr = dff_regr[:, trial_include_mask, :]
                # determine contra/ipsi side
                if self.data_index['hemisphere'][ii] == 'Right':
                    contra_action = 0
                    ipsi_action = 1
                else:
                    contra_action = 1
                    ipsi_action = 0
                
                # prepare the decoding variables
                # stimulus: 0/1 (for A/B, C/D)
                # actions: 0/1 (for ipsi/contra)
                if self.data_index['Protocol'][ii] == 'AB':
                    s = behDF['schedule'][trial_include_mask]-1 # A: -1; B: 1
                elif self.data_index['Protocol'][ii] == 'AB-CD':
                    s = behDF['schedule'][trial_include_mask]-3
                    s_AB = behDF['schedule'][~trial_include_mask]-1# for null distribution, A: -1, B: 1, C/D: 0
                
                if contra_action==1:
                    c = 1-behDF['actions'][trial_include_mask] 
                else:
                    c = behDF['actions'][trial_include_mask] 

                if self.data_index['Protocol'][ii] == 'AB-CD':
                    c_AB = c[~trial_include_mask]

                decodeVar = 'stimulus'
                # build the stratify variable
                stratifyVar = 2*s + c # 4 groups: A-ipsi, A-contra, B-ipsi, B-contra
                nShuffle = 1000
                decode_result = run_decoder(dff_regr, s, stratifyVar, 'SVC', nShuffle)

                # also randomly picking 10-100 neurons, and run decoding model
                nRepeats = 100
                nNeuronsPick = np.arange(10, min(150, nCells), 10)
                decode_result_subsample = np.full((len(nNeuronsPick), nTime, nRepeats), np.nan)  # to store results for different neuron numbers
                decode_result_subsample_shuffle = np.full((len(nNeuronsPick), nTime, nRepeats,nShuffle), np.nan)  # to store shuffle results for different neuron numbers
                for pIdx, nPick in tqdm(enumerate(nNeuronsPick)):
                    results = Parallel(n_jobs=-1)(
                        delayed(lambda seed: run_decoder(
                            dff_regr[:, :, np.random.default_rng(seed).choice(nCells, nPick, replace=False)],
                            s,
                            stratifyVar,
                            'SVC',
                            nShuffle
                        ))(pIdx * 1000 + i)
                        for i in range(nRepeats)
                        )   
                
                # unpack results
                    for nRep, result in enumerate(results):
                        decode_result_subsample[pIdx, :, nRep] = result['accuracy']
                        decode_result_subsample_shuffle[pIdx, :, nRep, :] = result['ctrl'].T
                    # for nRep in range(nRepeats):
                    # # randomly pick nPick neurons
                    #     neuron_idx = np.random.choice(nCells, nPick, replace=False)
                    #     decode_result = run_decoder(dff_regr[:,:,neuron_idx], s, stratifyVar, 'SVC', nShuffle)
                    #     decode_result_subsample[pIdx, :,nRep] = decode_result['accuracy']
                    #     decode_result_subsample_shuffle[pIdx, :,nRep,:] = decode_result['ctrl'].T


    def decoding_summary(self):
        # summarize the decoding result for each session
        pass

    def demixed_PCA(self):
        # demixed PCA
        pass

if __name__ == "__main__":

    # use interactive matplotlib backend
    import time

    plt.ion()

    root_dir = r'Y:\HongliWang\Miniscope\ASD'

    # odor behavior process
    t0 = time.time()
    Odor_Beh = BehDataOdor(root_dir)
    print(f"BehDataOdor: {time.time() - t0:.2f} sec")
    # rotarod behavior process

    t0 = time.time()
    imaging_Odor = Imaging(root_dir, Odor_Beh)
    print(f"Imaging init: {time.time() - t0:.2f} sec")

    t0 = time.time()
    imaging_Odor.dff_to_pickle()
    print(f"dff_to_pickle: {time.time() - t0:.2f} sec")
    # basic plots of calcium traces aligned to behavior
    # 1. PSTH
    #imaging_Odor.cal_traces()
    #imaging_Odor.auROC()
    #imaging_Odor.MLR_session()
    #imaging_Odor.decoding_session()
    #imaging_Odor.MLR_orthogonal()
    #imaging_Odor.GLM_time_kernel()


    #%% analysis to do
    # 1. go over longitudinal registration
    # 2. align df/f with behavior events
    rot_Beh = BehDataRotarod(root_dir)
    imaging_rotarod = Imaging(root_dir, rot_Beh)


    #%% check longitudinal registration result
    # look at contours
    # look at dFF traces
    Long_data = Longitudinal(root_dir, imaging_Odor, imaging_rotarod)
    Long_data.auROC_coregister()
# do auROC after Long_data registration, include the coregistered neurons analysis also
    