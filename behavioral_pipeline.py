# code to process behavior files and align calcium data with behavior timestamps
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from datetime import datetime
import re
from utils_imaging import *
from utils_beh import *
import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

import imageio.v3 as iio

import matlab.engine
eng = matlab.engine.start_matlab()
# add matlab code into the path
#eng.addpath(r'C:\Users\Linda\Documents\GitHub\ASD_RLWM\Behavior', nargout=0)

class BehData:

    def __init__(self, root_path):
        self.root_path = root_path
        self.data = os.path.join(self.root_path, 'Data')
        self.analysis = os.path.join(self.root_path, 'Analysis')
        self.summary = os.path.join(self.analysis, 'Summary')
        self.AnimalInfo = pd.read_csv(os.path.join(self.data, 'AnimalList.csv'))
        self.Animals = [str(x) for x in self.AnimalInfo['AnimalID']]
        self.Genotypes = self.AnimalInfo['Genotype']
        if 'Cells' in self.AnimalInfo.columns:
            self.ImageCell = self.AnimalInfo['Cells']
        else:
            self.ImageCell = [None] * len(self.Animals)
        if 'hemisphere' in self.AnimalInfo.columns:
            self.Hemisphere = self.AnimalInfo['hemisphere']
        else:   
            self.Hemisphere = [None] * len(self.Animals)


class BehDataOdor(BehData):

    def __init__(self, root_file):
        super().__init__(root_file)
        self.bodyparts = ['nose', 'head', 'left ear', 'right ear', 'left hand', 'right hand',
                          'spine 1', 'spine 2', 'spine 3', 'left foot', 'right foot', 'tail 1',
                          'tail 2', 'tail 3']
        self.make_dataIndex()
        self.behavior = 'Odor'

        # get the behCSV path
        self.load_data()
    
    def make_dataIndex(self):
        # Create a data index, each row is a session
        rows = []
        date_pattern = re.compile(r'(\d{8})')
        for aIdx, a in enumerate(self.Animals):
            animalFolder = os.path.join(self.data, a, 'Odor', 'Behavior')
            # get .mat files
            rawFiles = glob.glob(os.path.join(animalFolder, '*.mat'))

            files_by_date = defaultdict(list)

            for f in rawFiles:
                fname = os.path.basename(f)
                session = os.path.splitext(fname)[0]
                match = date_pattern.search(fname)
                date_str = match.group(1)
                # extract date (assumes YYYYMMDD somewhere at start)


                files_by_date[date_str].append(f)

            # create one row per date
            # count the protocol day
            protocol_day_counter = defaultdict(int)

            for date, behavior_paths in sorted(files_by_date.items()):

                if 'ABCD' in behavior_paths[0]:
                    protocol = 'AB-CD'
                elif 'AB' in behavior_paths[0]:
                    protocol = 'AB'

                protocol_day_counter[protocol] += 1
                pDay = protocol_day_counter[protocol]

                # check if imaging folder exist
                behRecordFolder = os.path.join(self.data, a, 'Odor', 'Imaging', date)
                if not os.path.exists(behRecordFolder):
                    ifRec = False
                    behRecordFolder = None
                else:
                    ifRec = True
                    
                    files = os.listdir(behRecordFolder)

                    csv_files = [
                        f for f in files
                        if f.endswith(".csv") and "DLC_" not in f and 'AITimeStamp' not in f
                    ]
                    raw_videos = [f for f in files 
                                  if f.endswith(".mp4") and "DLC_"  not in f]
                    DLC_files = [f for f in files
                                 if f.endswith(".csv") and 'DLC_' in f]
                    DLC_file = os.path.join(behRecordFolder, DLC_files[0])
                    behRecording = os.path.join(behRecordFolder, raw_videos[0])
                    behTimeStampPath = os.path.join(behRecordFolder, csv_files[0])
                    AIMatrixPath = glob.glob(os.path.join(behRecordFolder, '*_AITTL_*'))
                    AITimeStampPath = glob.glob(os.path.join(behRecordFolder, '*_AITimeStamp_*.csv'))

                extra_columns = {
                    'ROIFile': '',         # store as JSON
                    'behRecording': behRecording if ifRec else None,   # list of .mp4 files
                    'behTimeStamp': behTimeStampPath if ifRec else None,                # list of timestamps
                    'AIMatrix': AIMatrixPath[0] if ifRec else None,                    # list or array
                    'AITimsStamp': AITimeStampPath[0] if ifRec else None,                 # list or array
                    'DLC': DLC_file if ifRec else None,
                    'ImgTimeStamp': '',                # list or array
                    'ifCalImg': False,                 # boolean
                    'ifBehRecording': ifRec,            # boolean
                    'BehCSV': []
                }
                
                row_dict = {
                    'Animal': a,
                    'Genotype': self.Genotypes[aIdx],
                    'Date': date,
                    'Protocol': protocol,
                    'ProtocolDay': pDay,
                    'BehaviorPath': sorted(behavior_paths),   # LIST of .mat files
                    'AnalysisPath': os.path.join(self.analysis, a, 'Odor', 'Behavior', date)
                }

                            # Merge extra columns
                row_dict.update(extra_columns)

                rows.append(row_dict)


        self.data_index = pd.DataFrame(rows)

    def load_data(self):
        # Load behavior data from file
        # need to call matlab functions
        for bIdx, behFiles in enumerate(self.data_index['BehaviorPath']):
            csvPath = os.path.join(self.data_index['AnalysisPath'][bIdx], 
                                self.data_index['Date'][bIdx] + 
                                '_' + self.data_index['Protocol'][bIdx]+
                                str(self.data_index['ProtocolDay'][bIdx])+'.csv')
            if not os.path.exists(csvPath):

                results = []
                for beh in behFiles:
                    resultdf = eng.extract_behavior_df(beh)

                    # deal with float precision problem
                    resultdf['reward'] = resultdf['reward'].round(0)
                    resultdf['trial_types'] = resultdf['trial_types'].round(3)
                    resultdf['odors'] = resultdf['odors'].round(0)

                    results.append(resultdf)
                # save the csv file in analysis folder
                final_df =  pd.concat(results, ignore_index=True)

                if not os.path.exists(self.data_index['AnalysisPath'][bIdx]):
                    os.makedirs(self.data_index['AnalysisPath'][bIdx], exist_ok = True)

                final_df.to_csv(csvPath)
            
            self.data_index.loc[bIdx, 'BehCSV'] = csvPath


    def align_timeStamps(self):
        # align timestamps between behavior log and recording
        # similar from the align_timeStamps method in Imaging_pipeline
        # but without calcium time stamp
        # the AI timestamp tracks box 8-1 only - so when align sessions in box8-2
        # sessions in 8-1 on the same day need to be aligned first
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
                        # find the reference behavioral first (the one in box8-1)
                        id_box1 = re.search(r'ASD(\d+)_1_', self.data_index['AIMatrix'][ii]).group(1)
                        id_box2 = re.search(r'ASD(\d+)_2_', self.data_index['AIMatrix'][ii]).group(1)

                        # find the right AI channel
                        if id_box1 == self.data_index['Animal'][ii]:
                            AI_channel = 0
                        else:
                            AI_channel = 1
                        
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
                        is_high = AI_matrix[:,AI_channel] > 4
                        edges = np.diff(is_high.astype(int))
                        rising = np.where(edges == 1)[0] + 1
                        falling = np.where(edges == -1)[0] + 1
                        durations = (falling - rising) / AI_freq
                        # exclude durations longer than 0.2 seconds (manual valve opening)
                        valid_pulses = durations < 0.5
                        n_valid_events = np.sum(valid_pulses)

                        behDF = pd.read_csv(self.data_index['BehCSV'][ii])

                        # look for left correct trials
                        nLeftCorrect = np.sum(np.logical_and(behDF['schedule'] == 1, behDF['reward'] > 0))
                        
                        # make a plot, go over behDF, if a left choice reward = 3, count 3 high voltage event
                        # if a left choice reward = 2, count 2 high voltage event

                        # need to check if the first trial is reward size 2 in a rewardsize 3 trials
                        nR2 = np.sum(behDF['reward']==2)
                        nR3 = np.sum(behDF['reward']==3)
                        nPulses = np.sum(behDF['reward'][np.logical_or(behDF['schedule']==1, behDF['schedule']==3)])

                        if nR2 == 1 and nR3 > 1 and n_valid_events + 2 == nPulses: 
                            # if the first 2 openings was not logged
                            # some weird first trial bug (not sure what is the cause)
                            behDF_tt = behDF.iloc[1:,:].reset_index(drop=True)
                            # reset trial
                            behDF_tt['trial'] = np.arange(behDF_tt.shape[0])+1
                            behDF = behDF_tt
                            # save the new behDF
                            behDF.to_csv(self.data_index['BehCSV'][ii])
                            
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
                            # then align the time between two left reward trials
                            AI_tobe_aligned = AI_TS_interp[(AI_TS_interp/1000 >= t0_AI) & (AI_TS_interp/1000 < t1_AI)]
                            timestamps_tobe_aligned = len(AI_tobe_aligned)
                            
                            AI_TS_aligned[(AI_TS_interp/1000 >= t0_AI) & (AI_TS_interp/1000 < t1_AI)] = np.linspace(t0, t1, timestamps_tobe_aligned, endpoint=False)

                    #%% based on the alignment betweeen AI_TS_interp and AI_TS_aligned, align behTimeStamp and ImgTimeStamp
                    # load behavior recording timestamp if exists
                    # check if it is aligned
                    old_path = self.data_index['behTimeStamp'][ii]
                    folder, old_file = os.path.split(old_path)
                    new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                    if not os.path.exists(new_file):
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

                        self.data_index.loc[ii,'behTimeStamp_aligned'] = new_file
                        behTimeStamp.to_csv(new_file, index=False)
     

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
                    self.data_index.loc[ii,'behTimeStamp_aligned'] = new_file 


    def session_behavior(self):
        """ it is probably way easier to do it just in Matlab"""
        # plot behavior of each individual session
        nSessions = self.data_index.shape[0]
        for ss in range(nSessions):
            resultdf_path = self.data_index['BehCSV'][ss]
            #resultdf = resultdf.drop(columns=['Unnamed: 0'], errors='ignore')
            #data_dict = resultdf.to_dict(orient='list')
            
            eng.ASD_session(resultdf_path,self.data_index['Protocol'][ss],self.data_index['Animal'][ss], 
                            self.data_index['Date'][ss],self.data_index['AnalysisPath'][ss],nargout=0)

    def odor_summary(self):
        pass

    def performance(self):
        # call matlab function to plot the performance
        pass

    def model_fit(self):
        # call matlab function to fit the computational model
        pass

    def DLC_analysis(self):
        # analyze DLC result per session
        # speed, position, head direction, average trajectory aligned to center_in

        # plot the trajectory around center_in and side_in
        # plot the speed around center_in and side_in

        # load DLC result, make some plot
        nFiles = self.data_index.shape[0]

        if self.behavior == 'Odor':
            for ii in range(nFiles):
                # check if figure has been generated
                savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                    self.data_index['Date'][ii])
                DLCPath = self.data_index['DLC'][ii]
                DLCdata = load_DLC(DLCPath)

                behDF = pd.read_csv(self.data_index['BehCSV'][ii])

                # align body parts to center_in and side_in\
                # smooth it to remove jumping parts
                nTrials = behDF.shape[0]
                bodyparts = DLCdata['bodyparts']
                aligned_keypoints = {}
                startTime = -1.99
                endTime = 1.99
                aligned_t = np.arange(startTime, endTime,0.02)

                # video timestamp
                videoTS = pd.read_csv(self.data_index['behTimeStamp_aligned'][ii], header=0)
                header = ['TimeStamp', 'AlignedTimeStamp']
                videoTS.columns = header

                align_events = ['center_in', 'side_in']
                for event in align_events:
                    aligned_keypoints[event] = {}
                    for bp in bodyparts:
                        aligned_keypoints[event][bp] = {}
                        aligned_keypoints[event][bp]['x'] = np.full((len(aligned_t), nTrials), np.nan)
                        aligned_keypoints[event][bp]['y'] = np.full((len(aligned_t), nTrials), np.nan)
                        smoothed_x = moving_average(DLCdata[bp]['x'], window=10)
                        smoothed_y = moving_average(DLCdata[bp]['y'], window=10)
                        for tt in range(nTrials):
                            
                        # look for the time, interpolate it
                            t_middle = behDF[event][tt]
                            if not np.isnan(t_middle): # could be nan for missed trials in side_in   
                                t_start = t_middle + startTime
                                t_end = t_middle + endTime
                                timeMask = np.logical_and(videoTS['AlignedTimeStamp']<t_end, 
                                                    videoTS['AlignedTimeStamp']>t_start)
                                center_kp_x = smoothed_x[timeMask]
                                center_kp_y = smoothed_y[timeMask]
                                sig_t = videoTS['AlignedTimeStamp'][timeMask]-t_middle

                                aligned_keypoints[event][bp]['x'][:,tt] = np.interp(aligned_t, sig_t, center_kp_x)
                                aligned_keypoints[event][bp]['y'][:,tt] = np.interp(aligned_t, sig_t, center_kp_y)

                choice = np.array(behDF['actions'])
                prev_choice = np.concatenate([[np.nan], choice[0:-1]])
                plt.figure()
                plt.imshow(frame)
                left_choice_trials = choice==0
                left_choice_prev = prev_choice==0
                right_choice_prev = prev_choice==1
                right_choice_trials = choice == 1
                X= aligned_keypoints['center_in']['head']['x'][:,(left_choice_trials & left_choice_prev)]
                Y= aligned_keypoints['center_in']['head']['y'][:,(left_choice_trials & left_choice_prev)]
                # --- plot single trials ---
                # for tt in range(nTrials):
                #     plt.plot(X[:, tt], Y[:, tt],
                #             color='gray', linewidth=1, alpha=0.5)

                # --- compute average trajectory ---
                x_mean = np.nanmean(X, axis=1)
                y_mean = np.nanmean(Y, axis=1)
                x_std = np.nanstd(X, axis=1)
                y_std = np.nanstd(Y, axis=1)

                # --- plot average ---
                plt.plot(x_mean, y_mean,
                        color='red', linewidth=3, label='mean')
                for k in range(-1, 2):
                    plt.plot(x_mean + k * x_std,
                            y_mean + k * y_std,
                            color='red',
                            alpha=0.2)
                # plot right-left trials
                X= aligned_keypoints['center_in']['head']['x'][:,(left_choice_trials & right_choice_prev)]
                Y= aligned_keypoints['center_in']['head']['y'][:,(left_choice_trials & right_choice_prev)]

                x_mean = np.nanmean(X, axis=1)
                y_mean = np.nanmean(Y, axis=1)

                # --- plot average ---
                plt.plot(x_mean, y_mean,
                        color='blue', linewidth=3, label='mean')

                plt.plot(np.nanmean(aligned_keypoints['center_in']['head']['x'][:,(right_choice_trials & left_choice_prev)], axis=1), 
                         np.nanmean(aligned_keypoints['center_in']['head']['y'][:,(right_choice_trials & left_choice_prev)], axis=1),
                        color='yellow', linewidth=3, label='mean')
                
                plt.plot(np.nanmean(aligned_keypoints['center_in']['head']['x'][:,(right_choice_trials & right_choice_prev)], axis=1), 
                         np.nanmean(aligned_keypoints['center_in']['head']['y'][:,(right_choice_trials & right_choice_prev)], axis=1),
                        color='green', linewidth=3, label='mean')
                
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Trajectory (single trials + mean)')
                plt.axis('equal')
                plt.legend()

                plt.show()

                
                plt.figure()
                plt.imshow(frame)
                sc = plt.scatter(aligned_keypoints['center_in']['head']['x'][:,0], aligned_keypoints['center_in']['head']['y'][:,0],
                                  c=aligned_t, cmap='viridis', s=10)

                plt.colorbar(sc, label='Time (s)')
                plt.scatter(aligned_keypoints['center_in']['head']['x'][100,0], aligned_keypoints['center_in']['head']['y'][100,0], s=40)
                x_smooth = moving_average(DLCdata['head']['x'], window=10)
                y_smooth = moving_average(DLCdata['head']['y'], window=10)

                # plot head position near the center_in time
                nTrials = behDF.shape[0]
                center_head_x = []
                center_head_y = []

                center_head_x_smoothed = []
                center_head_y_smoothed = []
                for tt in range(nTrials):
                    center_in = behDF['center_in'][tt]
                    center_out = behDF['center_out'][tt]
                    timeMask = np.logical_and(videoTS['AlignedTimeStamp']<center_out, 
                                              videoTS['AlignedTimeStamp']>center_in)
                    center_head_x.append(np.array(DLCdata['head']['x'])[timeMask])
                    center_head_y.append(np.array(DLCdata['head']['y'])[timeMask])
                    center_head_x_smoothed.append(x_smooth[timeMask])
                    center_head_y_smoothed.append(y_smooth[timeMask])


                center_x = np.concatenate(center_head_x)
                center_y = np.concatenate(center_head_y)
                center_x_smoothed = np.concatenate(center_head_x_smoothed)
                center_y_smoothed = np.concatenate(center_head_y_smoothed)


                videopath = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\Data\578\Odor\Imaging\20251216\ASD578__2025-12-16T11_56_01.mp4'
                ts = videoTS['AlignedTimeStamp'].to_numpy()
                target = behDF['center_in'][0]

                idx = np.argmin(np.abs(ts - target))
                frame = iio.imread(videoPath, index=339000)
                tsFile = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\599\Odor\Imaging\20260202\ASD599__2026-02-02T13_22_50.csv'
                ts = pd.read_csv(tsFile)
                header = ['TimeStamp']
                ts.columns = header
                
                import subprocess

                def trim_by_time(input_file, output_file, start_sec, end_sec):
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-ss", str(start_sec),
                        "-to", str(end_sec),
                        "-i", input_file,
                        "-c:v", "libx264",   # re-encode for accuracy (important for VFR)
                        "-c:a", "aac"
                    ]
                    subprocess.run(cmd, check=True)
                output = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\599\Odor\Imaging\20260202\ASD599__2026-02-02T13_22_50_trimmed.mp4'
                trim_by_time(videoPath, output, 0, 11832.6)

                # frame is a numpy array (H x W x 3)
                print(frame.shape)


class BehDataRotarod(BehData):

    def __init__(self, root_file):
        super().__init__(root_file)
        self.make_dataIndex()

        self.load_data()
        self.behavior = 'Rotarod'

    def make_dataIndex(self):
        # Create a data index, each row is a session
        DLC_results = []
        stage = []
        Rod_speed = []
        timeStamp = []
        video= []
        animalID = []
        analysis = []
        Genotype = []
        sessionID = []
        #sexID = []
        dateList = []
        timeOnRod = []
        fallbyTurning = []

        # %% load all files
        for aidx,aa in enumerate(self.Animals):
            dataFolder = os.path.join(self.data, aa, 'Rotarod', 'Behavior')
            dateFolder = sorted([f for f in os.listdir(dataFolder) if os.path.isdir(os.path.join(dataFolder, f))])
            filePatternSpeed = aa + '*speed*.csv'
            filePatternDLC = '*ASD' + aa + '*.csv'
            filePatternVideo = aa + '*.avi'
            filePatternTimestamp = aa + '*timeStamp*.csv'
            rr_results_path = os.path.join(self.data,aa, 'Rotarod','Behavior', 'RR_results.csv')
            rr_results = pd.read_csv(rr_results_path)
            for date in dateFolder:
                speedCSV = glob.glob(os.path.join(dataFolder, date, filePatternSpeed))
                timeStampCSV = glob.glob(os.path.join(dataFolder, date, filePatternTimestamp))
                videoFiles = glob.glob(os.path.join(dataFolder, date, filePatternVideo))
                DLCFiles = glob.glob(os.path.join(dataFolder, date, filePatternDLC))
                num_files = len(videoFiles)

                if num_files>0:
                    for ff in range(num_files):
                        # match the sessions
                        dateExpr = r'\d{6}_trial\d{1,2}'
                        matches = re.findall(dateExpr,videoFiles[ff][0:-23])
                        # in tempVideo['back'], find the string that has matches
                        video.append(videoFiles[ff])
                        DLC_ID = [ID for ID in range(len(DLCFiles)) if matches[0] in DLCFiles[ID]]
                        if len(DLC_ID)>0:
                            DLC_results.append(DLCFiles[DLC_ID[0]])
                        else:
                            DLC_results.append(None)
                        speed_ID = [ID for ID in range(len(speedCSV)) if matches[0] in speedCSV[ID]]
                        Rod_speed.append(speedCSV[speed_ID[0]])
                        timeStamp_ID = [ID for ID in range(len(timeStampCSV)) if matches[0] in timeStampCSV[ID]]
                        timeStamp.append(timeStampCSV[timeStamp_ID[0]])

                        animalID.append(aa)

                        #stage.append(matches[0])
                        analysis.append(os.path.join(self.analysis, aa,'Rotarod', 'Behavior', matches[0]))
                        ses = re.search(r'\d{1,2}\s*$', matches[0])
                        sessionID.append(int(ses.group()))
                        dateList.append(matches[0][0:6])
                        Genotype.append(self.Genotypes[aidx])
                        #sexID.append(self.Sex[aidx])

                        # find the animal and trial in rr_result
                        result = rr_results[(rr_results['Trial'] == int(ses.group()))]
                        timeOnRod.append(int(result['Time'].values[0]))
                        fallbyTurning.append(result['fall by turning'].astype(bool).values[0])

        self.data_index = pd.DataFrame(animalID, columns=['Animal'])
        self.data_index['DLC'] = DLC_results
        self.data_index['Video'] = video
        self.data_index['Rod_speed'] = Rod_speed
        self.data_index['AnalysisPath'] = analysis
        self.data_index['Genotype'] = Genotype
        #self.data['Sex'] = sexID
        self.data_index['Trial'] = sessionID
        self.data_index['Date'] = dateList
        self.data_index['BehTimestamp'] = timeStamp
        self.data_index['TimeOnRod'] = timeOnRod
        self.data_index['FallByTurning'] = fallbyTurning

        self.nSubjects = len(self.Animals)
        #sorted_df = self.dataIndex.sort_values(by=['Animal', 'Trial'])
        #sorted_df = sorted_df.reset_index(drop=True)
        #self.data=sorted_df
        #self.nSessions = len(self.data['Animal'])

    def load_data(self):
        # Load rotarod behavior data from file
        pass
    

    def align_with_calcium(self, calcium_timestamps):
        # Align rotarod behavior timestamps with calcium imaging timestamps
        pass


if __name__ == "__main__":
    root_dir = r'Y:\HongliWang\Miniscope\ASD'

    #%% test code for odor behavior
    Odor = BehDataOdor(root_dir)

    # #%% load matlab code
    #eng = matlab.engine.start_matlab()

    # code_folder = r'C:\Users\Linda\Documents\GitHub\ASD_RLWM'
    #eng.addpath(eng.genpath(code_folder), nargout=0)

    # # read the data and save them to csv files
    Odor.load_data()

    #Odor.session_behavior()

    #%% for behavior recordings
    # run it separately, if calcium imaging exist, then run the align_timeStamps in Imaging_pipeline


    #%% test code for rotarod behavior
    rotarod = BehDataRotarod(root_dir)

    rotarod.load_data()