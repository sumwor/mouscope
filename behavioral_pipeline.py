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
        self.summary = os.path.join(self.root_path, 'Summary')
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
                if 'CD' in behavior_paths[0] and 'DC' in behavior_paths[0]:
                    protocol = 'AB-CD-DC'
                elif 'CD'in behavior_paths[0] and not 'DC' in behavior_paths[0]:
                    protocol = 'AB-CD'
                elif 'DC' in behavior_paths[0] and not 'CD' in behavior_paths[0]:
                    protocol = 'AB-DC'
                else:
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

    def plot_performance(self):
        # call matlab function to plot the performance

        perf_df = pd.DataFrame(columns=['Animal', 'Genotype', 'Date', 'Protocol', 'ProtocolDay', 'RewardRate', 'd'])
        for bIdx, behFiles in enumerate(self.data_index['BehCSV']):
            # load the files, calculate performance in 100-trial blocks
            resultdf = pd.read_csv(behFiles)
            perf_df.loc[bIdx, 'Animal'] = self.data_index['Animal'][bIdx]
            perf_df.loc[bIdx, 'Genotype'] = self.data_index['Genotype'][bIdx]
            perf_df.loc[bIdx, 'Date'] = self.data_index['Date'][bIdx]
            perf_df.loc[bIdx, 'Protocol'] = self.data_index['Protocol'][bIdx]
            perf_df.loc[bIdx, 'ProtocolDay'] = self.data_index['ProtocolDay'][bIdx]
            perf_df.loc[bIdx, 'RewardRate'] = np.full((25,1), np.nan)
            perf_df.loc[bIdx, 'd'] = np.full((25,1), np.nan)

            nTrials = resultdf.shape[0]
            tBlocks = 100
            
            protocol = self.data_index['Protocol'][bIdx]
            # determine the session length
            if protocol == 'AB':
                startTrial = 0
                sti_A = 1
                sti_B = 2
            elif protocol == 'AB-CD':
                # look for the first trial with schedule 3 or 4
                startTrial = np.where(resultdf['schedule']>=3)[0][0]
                sti_A = 3
                sti_B = 4
            elif protocol == 'AB-DC' or protocol == 'AB-CD-DC':
                startTrial = np.where(resultdf['schedule']>=5)[0][0]
                sti_A = 5
                sti_B = 6
            endTrial = nTrials
            result = resultdf.iloc[startTrial:endTrial,:].reset_index(drop=True)
            nBlocks = result.shape[0] // tBlocks
                # look for the first trial with schedule 3 or 4, and the first trial with schedule 2 or 4

            for bb in range(nBlocks):

                block_df = result.iloc[bb*tBlocks:(bb+1)*tBlocks,:]
                # average reward rate
                perf = np.sum(block_df['reward']>0)/tBlocks
                perf_df.loc[bIdx, 'RewardRate'][bb] = perf

                # d_prime 
                # hit rate P(right | B)
                # false alarm rate P(right | A)

                hit_rate = np.sum((block_df['actions']==1) & (block_df['schedule']==sti_B))/np.sum(block_df['schedule']==sti_B)
                false_alarm_rate = np.sum((block_df['actions']==1) & (block_df['schedule']==sti_A))/np.sum(block_df['schedule']==sti_A)
                if hit_rate == 1:
                    hit_rate = 0.99999
                if false_alarm_rate == 1:
                    false_alarm_rate = 0.99999
                if hit_rate == 0:
                    hit_rate = 0.00001
                if false_alarm_rate == 0:
                    false_alarm_rate = 0.00001
                d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
                perf_df.loc[bIdx, 'd'][bb] = d_prime

        # plot average performance for AB1, AB2, AB3
        # CD1, CD2, and CD3. and run stats
        # rebuild performance dataframe for plotting
        perf_plot_AB1 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_AB2 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_AB3 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD1 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD2 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD3 = pd.DataFrame(columns=['Animal', 'Genotype', 'Block', 'RewardRate', 'd'])
        for idx, row in perf_df.iterrows():
            for bb in range(10):
                if not np.isnan(row['RewardRate'][bb]):
                    if row['Protocol'] == 'AB':
                        if row['ProtocolDay'] == 1:   
                            perf_plot_AB1 = pd.concat([perf_plot_AB1, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 2:
                            perf_plot_AB2 = pd.concat([perf_plot_AB2, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 3:
                            perf_plot_AB3 = pd.concat([perf_plot_AB3, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                    elif row['Protocol'] == 'AB-CD':
                        if row['ProtocolDay'] == 1:   
                            perf_plot_CD1 = pd.concat([perf_plot_CD1, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb-12, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 2:
                            perf_plot_CD2 = pd.concat([perf_plot_CD2, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb-12, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 3:
                            perf_plot_CD3 = pd.concat([perf_plot_CD3, pd.DataFrame([{'Animal': row['Animal'], 'Genotype': row['Genotype'], 'Block': bb-12, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
        plot_learning_curve(perf_plot_AB1, save_name = 'AB1_rewardrate', 
                            value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary)
        plot_learning_curve(perf_plot_AB2, save_name = 'AB2_rewardrate', 
                    value_col = 'RewardRate', trial_col = 'Block')
        plot_learning_curve(perf_plot_AB3, save_name = 'AB3_rewardrate',
                    value_col = 'RewardRate', trial_col = 'Block')
        plot_learning_curve(perf_plot_CD1, save_name = 'CD1_rewardrate',
                    value_col = 'RewardRate', trial_col = 'Block')
        plot_learning_curve(perf_plot_CD2, save_name = 'CD2_rewardrate',
                    value_col = 'RewardRate', trial_col = 'Block')
        plot_learning_curve(perf_plot_CD3, save_name = 'CD3_rewardrate',
                    value_col = 'RewardRate', trial_col = 'Block')
        
        # plot_learning_curve(perf_plot_AB1, save_name = 'AB1_dprime', 
        #     value_col = 'd', trial_col = 'Block')
        # plot_learning_curve(perf_plot_AB2, save_name = 'AB2_dprime', 
        #     value_col = 'd', trial_col = 'Block')
        
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


                # videopath = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\Data\578\Odor\Imaging\20251216\ASD578__2025-12-16T11_56_01.mp4'
                # ts = videoTS['AlignedTimeStamp'].to_numpy()
                # target = behDF['center_in'][0]

                # idx = np.argmin(np.abs(ts - target))
                # frame = iio.imread(videoPath, index=339000)
                # tsFile = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\599\Odor\Imaging\20260202\ASD599__2026-02-02T13_22_50.csv'
                # ts = pd.read_csv(tsFile)
                # header = ['TimeStamp']
                # ts.columns = header
                
                # import subprocess

                # def trim_by_time(input_file, output_file, start_sec, end_sec):
                #     cmd = [
                #         "ffmpeg",
                #         "-y",
                #         "-ss", str(start_sec),
                #         "-to", str(end_sec),
                #         "-i", input_file,
                #         "-c:v", "libx264",   # re-encode for accuracy (important for VFR)
                #         "-c:a", "aac"
                #     ]
                #     subprocess.run(cmd, check=True)
                # output = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\599\Odor\Imaging\20260202\ASD599__2026-02-02T13_22_50_trimmed.mp4'
                # trim_by_time(videoPath, output, 0, 11832.6)

                # # frame is a numpy array (H x W x 3)
                # print(frame.shape)


class BehDataRotarod(BehData):

    def __init__(self, root_file):
        super().__init__(root_file)
        self.make_dataIndex()

        self.behavior = 'Rotarod'

    def make_dataIndex(self):
        # Create a data index, each row is a session
        # build the data index from rr_result
        rr_results_path = os.path.join(self.data, 'RR_results.csv')
        rr_results = pd.read_csv(rr_results_path)

        self.data_index = pd.DataFrame()
        self.data_index['Animal'] = rr_results['AnimalID']
        self.data_index['Trial'] = rr_results['Trial']
        self.data_index['Date'] = rr_results['Date']
        self.data_index['Performance'] = rr_results['Performance']
        self.data_index['FallByTurning'] = rr_results['FBT']
        self.data_index['Genotype'] = rr_results['Genotype']

        # make a list of DLC result, rod speed, timestamp, video path, and analysis path for each session
        # length should be the same as the number of rows in data_index

        DLC_results = [[] for _ in range(self.data_index.shape[0])]
        Rod_speed = [[] for _ in range(self.data_index.shape[0])]
        timeStamp = [[] for _ in range(self.data_index.shape[0])]
        video= [[] for _ in range(self.data_index.shape[0])]
        analysis = [[] for _ in range(self.data_index.shape[0])]


        # %% load all files
        for aidx in range(self.data_index.shape[0]):
            aa = self.data_index['Animal'][aidx]
            dataFolder = os.path.join(self.data, aa, 'Rotarod', 'Behavior')
            if os.path.exists(dataFolder):
                dateFolder = sorted([f for f in os.listdir(dataFolder) if os.path.isdir(os.path.join(dataFolder, f))])
                filePatternSpeed = aa + '*speed*.csv'
                filePatternDLC = aa + '*DLC_resnet*.csv'
                filePatternVideo = aa + '*.avi'
                filePatternTimestamp = aa + '*timeStamp*.csv'

                for date in dateFolder:
                    speedCSV = glob.glob(os.path.join(dataFolder, date, filePatternSpeed))
                    timeStampCSV = glob.glob(os.path.join(dataFolder, date, filePatternTimestamp))
                    videoFiles = glob.glob(os.path.join(dataFolder, date, filePatternVideo))
                    DLCFiles = glob.glob(os.path.join(dataFolder, date, filePatternDLC))
                    num_files = len(videoFiles)

                    if num_files>0:
                        for ff in range(num_files):
                            # match the sessions: ASDxxx followed by optional middle part, then trialx(x), optional underscore, and date YYYY-MM-DDTHH...
                            dateExpr = r'ASD\d+.*?_trial\d{1,2}(?=_?\d{4}-\d{2}-\d{2}T)'
                            matches = re.findall(dateExpr,videoFiles[ff])
                            # in tempVideo['back'], find the string that has matches
                            video[aidx] = videoFiles[ff]
                            DLC_ID = [ID for ID in range(len(DLCFiles)) if matches[0] in DLCFiles[ID]]
                            if len(DLC_ID)>0:
                                DLC_results[aidx] = DLCFiles[DLC_ID[0]]
                            else:
                                DLC_results[aidx] = None
                            speed_ID = [ID for ID in range(len(speedCSV)) if matches[0] in speedCSV[ID]]

                            if len(speed_ID)>0:
                                Rod_speed[aidx] = speedCSV[speed_ID[0]]
                            else:
                                Rod_speed[aidx] = None

                            timeStamp_ID = [ID for ID in range(len(timeStampCSV)) if matches[0] in timeStampCSV[ID]]
                            timeStamp[aidx] = timeStampCSV[timeStamp_ID[0]]


                            #stage.append(matches[0])
                            analysis[aidx] = os.path.join(self.analysis, aa,'Rotarod', 'Behavior', matches[0])



        self.data_index['DLC'] = DLC_results
        self.data_index['Video'] = video
        self.data_index['Rod_speed'] = Rod_speed
        self.data_index['AnalysisPath'] = analysis
        self.data_index['BehTimestamp'] = timeStamp


        self.nSubjects = len(self.Animals)
        #sorted_df = self.dataIndex.sort_values(by=['Animal', 'Trial'])
        #sorted_df = sorted_df.reset_index(drop=True)
        #self.data=sorted_df
        #self.nSessions = len(self.data['Animal'])

    def plot_performance(self):
        perf_df = self.data_index[['Animal', 'Genotype', 'Trial', 'Performance', 'FallByTurning']].copy()
        perf_df['Performance'] = pd.to_numeric(perf_df['Performance'], errors='coerce')
        if perf_df['FallByTurning'].dtype == bool:
            fbt_mask = perf_df['FallByTurning'].fillna(False)
        else:
            fbt_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
            fbt_text = perf_df['FallByTurning'].astype(str).str.lower().isin(['true', '1', 'yes'])
            fbt_mask = fbt_numeric | fbt_text
        perf_df.loc[fbt_mask, 'Performance'] = np.nan
        perf_df = perf_df.rename(columns={
            'Animal': 'subject',
            'Genotype': 'genotype',
            'Trial': 'trial',
            'Performance': 'time_on_rod'
        })
        perf_df['trial'] = pd.Categorical(perf_df['trial'], categories=np.sort(perf_df['trial'].dropna().unique()), ordered=True)
        genotype_order = [g for g in ['WT', 'HET', 'KO'] if g in set(perf_df['genotype'].dropna())]
        genotype_order += [g for g in perf_df['genotype'].dropna().unique() if g not in genotype_order]
        perf_df['genotype'] = pd.Categorical(perf_df['genotype'], categories=genotype_order)
        genotype_counts = perf_df.dropna(subset=['subject', 'genotype']).groupby('genotype', observed=True)['subject'].nunique()

        os.makedirs(self.summary, exist_ok=True)
        perf_df.to_csv(os.path.join(self.summary, 'Rotarod performance.csv'), index=False)

        stats_df,_,_ = run_learning_gam(perf_df, summary_path = self.summary)

        # ------------------------------------------------------------
        # Robustly mark failed trials (FallByTurning)
        # ------------------------------------------------------------
        exclude_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
        exclude_text = perf_df['FallByTurning'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )

        perf_df.loc[exclude_numeric | exclude_text, 'time_on_rod'] = np.nan

        # ------------------------------------------------------------
        # Clean data
        # ------------------------------------------------------------
        clean_df = perf_df.dropna(
            subset=['subject', 'genotype', 'trial', 'time_on_rod']
        ).copy()

        summary_df = clean_df.groupby(['genotype', 'trial'], observed=True)['time_on_rod'].agg(['mean', 'std', 'count']).reset_index()
        summary_df['sem'] = summary_df['std'] / np.sqrt(summary_df['count'])
        trial_codes = {trial: idx for idx, trial in enumerate(perf_df['trial'].cat.categories)}
        colors = {'WT': 'black', 'HET': 'red', 'KO': 'red'}
        stats_text = None
        if not stats_df.empty and 'p_value' in stats_df.columns:
            p_values = stats_df.set_index('term')['p_value']
            stats_text = (
                f"GAM p genotype = {p_values.get('genotype', np.nan):.3g}\n"
                f"GAM p learning = {p_values.get('learning', np.nan):.3g}\n"
                f"GAM p interaction = {p_values.get('genotype:learning', np.nan):.3g}"
            )

        fig, ax = plt.subplots(figsize=(7, 5))
        for genotype in genotype_order:
            genotype_data = summary_df[summary_df['genotype'] == genotype]
            if genotype_data.empty:
                continue
            x = genotype_data['trial'].map(trial_codes).astype(float).to_numpy()
            y = genotype_data['mean'].to_numpy(dtype=float)
            sem = genotype_data['sem'].to_numpy(dtype=float)
            color = colors.get(genotype, None)
            label = f'{genotype} (n={genotype_counts.get(genotype, 0)})'
            ax.errorbar(x, y, yerr=sem, marker='o', linewidth=2, capsize=3, color=color, label=label)

            raw = clean_df[clean_df['genotype'] == genotype]
            raw_x = raw['trial'].map(trial_codes).astype(float).to_numpy()
            jitter = (genotype_order.index(genotype) - (len(genotype_order) - 1) / 2) * 0.08
            ax.scatter(raw_x + jitter, raw['time_on_rod'], s=18, alpha=0.35, color=color, edgecolors='none')

        ax.set_xticks(np.arange(len(trial_codes)))
        ax.set_xticklabels([str(t) for t in trial_codes.keys()])
        ax.set_xlabel('Trial')
        ax.set_ylabel('Time on rod (s)')
        ax.set_title('Rotarod Performance')
        ax.legend(frameon=False)
        if stats_text is not None:
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.summary, 'Rotarod performance.png'), dpi=300)
        plt.savefig(os.path.join(self.summary, 'Rotarod performance.svg'), format='svg')
        plt.close(fig)

        return perf_df, stats_df
    def load_data(self):
        # Load rotarod behavior data from file
        DLC_obj= []

        for s in range(self.nSessions):
            analysisPath = self.data['AnalysisPath'][s]

            filePath = self.data['DLC'][s]
            videoPath = self.data['Video'][s]
            rodPath = self.data['Rod_speed'][s]
            fps = self.data['Timestamp'][s]
            dlc = DLCSession(filePath, videoPath, rodPath, analysisPath, fps)
            DLC_obj.append(dlc)

        self.data['DLC_obj'] = DLC_obj
        #self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSessions)
        self.WTIdx = animalIdx[self.data['GeneBG'] == groups[0]]
        self.MutIdx = animalIdx[self.data['GeneBG'] == groups[1]]
        # grouping the animals

        if self.Sex[0]==np.nan: # if no sex info
            nGroups = 1
        else:
            nGroups = 2

        if nGroups==2:
            self.maleIdx = np.where(self.data['Sex']=='M')[0]
            self.femaleIdx = np.where(self.data['Sex']=='F')[0]

        self.startVoltage = [4.45, 40] # 5 rpm = 0.273 V
        self.endVoltage = [8.90, 80]
        self.rod_a = (self.endVoltage[1] - self.startVoltage[1]) / (self.endVoltage[0] - self.startVoltage[0])
        self.rod_b = self.endVoltage[1] - self.endVoltage[0] * self.rod_a

    

    def align_with_calcium(self, calcium_timestamps):
        # Align rotarod behavior timestamps with calcium imaging timestamps
        pass

    def get_stride(self):        
        savedatapath = os.path.join(self.analysis, 'stride_freq.csv')
        runFile = os.path.join(self.analysis, 'notExist.csv') # a not existing file to allow re-calculating
        if not os.path.exists(runFile):

            savefigpath = os.path.join(self.analysis)
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)

            # get rid the the turning period
            
            # %% define rod plane first
            # load reference point
            ave_left_rod_back = self.data['left_rod_back']
            ave_right_rod_back = self.data['right_rod_back']
            ave_center_rod_back = self.data['center_rod_back']
            ave_left_rod_front = self.data['left_rod_front']
            ave_right_rod_front = self.data['right_rod_front']
            ave_center_rod_front = self.data['center_rod_front']

            ref_plot = os.path.join(savefigpath, 'Rod coordinate.png')
            if not os.path.exists(ref_plot):
                frame = read_video(self.videoPath, 0, ifgray=False)
                # overlay the video frame?
                plt.figure()
                plt.imshow(frame)
                plt.scatter(self.data['rod_left_back']['x'], self.data['rod_left_back']['y'],
                            c=self.data['rod_left_back']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood
                plt.colorbar(label='Confidence')

                plt.scatter(self.data['rod_right_back']['x'], self.data['rod_right_back']['y'],
                            c=self.data['rod_right_back']['p'], cmap='viridis', s=100)

                plt.scatter(self.data['rod_left_front']['x'], self.data['rod_left_front']['y'],
                            c=self.data['rod_left_front']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood

                plt.scatter(self.data['rod_right_front']['x'], self.data['rod_right_front']['y'],
                            c=self.data['rod_right_front']['p'], cmap='viridis', s=100)

                # get average from keypoints with confidence higher than 95


                plt.scatter(ave_left_rod_back[0],ave_left_rod_back[1], s=500)
                plt.scatter(ave_right_rod_back[0], ave_right_rod_back[1], s=500)
                plt.scatter(ave_center_rod_back[0], ave_center_rod_back[1], s=500)

                plt.scatter(ave_left_rod_front[0],ave_left_rod_front[1], s=500)
                plt.scatter(ave_right_rod_front[0], ave_right_rod_front[1], s=500)
                plt.scatter(ave_center_rod_front[0], ave_center_rod_front[1], s=500)

                plt.savefig(os.path.join(savefigpath, 'rod_plane.png'))
                plt.close()
            # save the figure

            # %% examine the body parts in back view and front view

            # find behavior time (from rod start turning to fall)
            startTime= self.data['rodT'][self.data['rodSpeed_smoothed']>0][0]
            if np.isnan(self.data['rodStart'][0]):
                self.data['rodStart'][0] = 0
            endTime = startTime+df_entry['TimeOnRod'] + self.data['rodRun'][0] - self.data['rodStart'][0] # need the time stay on rod

            timeMaskDLC = np.logical_and(self.data['time']>=startTime, self.data['time']<= endTime)
            timeMaskRod = np.logical_and(self.data['rodT']>=startTime, self.data['rodT']<= endTime)
            nFramesInclude = np.sum(timeMaskDLC)
            time_include = self.data['time'][timeMaskDLC]
            kp_list = ['left hand', 'right hand', 'left foot', 'right foot']
            self.stride = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_front[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_front[1])**2)
                elif 'foot' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_back[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_back[1])**2)

            # try calculate the stride using distance from the rod (a horizontal line)
            self.stride_rod = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_left_rod_front, ave_right_rod_front)
                elif 'foot' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_right_rod_back, ave_left_rod_back)

            #tempMask = ~p_mask[timeMask]
            #    self.stride[tempMask,idx] = np.nan
            #    tempStride, tempIdx = fill_nans_and_split(self.stride[:, idx])
                # interpolate the nans
                # self.notnanChunks[kp] = tempIdx
                # for ich, chunk in enumerate(tempIdx):
                #     self.stride[chunk[0]:chunk[1]+1,idx] = tempStride[ich]


            # low-pass filter the data
            fps = 50
            self.t_interp = np.arange(time_include[0], time_include[-1] + 1 / fps, 1 / fps)
            self.filtered_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)
            self.interp_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)

            # need to determine the cutoff frequency here
            for idx, kp in enumerate(kp_list):
                #for ich, chunk in enumerate(self.notnanChunks[kp]):
                #    if chunk[1]-chunk[0]+1 > 18:  # padlen
                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
                self.interp_stride[:,idx] = np.interp(self.t_interp, time_include, self.stride_rod[:,idx])

                self.filtered_stride[:,idx] = butter_lowpass_filter(self.interp_stride[:,idx], 5,fps,order=5)

            #%%
            # examine the autocorrelation
            # average them over genotype and trial
            # find the time when rod speed reach 5/10
            #if df_entry['Trial']<=6:
            startSpeed = 5
            #else:
            #    startSpeed = 10
            startTime_auto = self.data['rodT'][self.data['rodSpeed_smoothed']>startSpeed][0]
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 subplots
            ax = ax.flatten()
            for ss in range(len(kp_list)):
                signal = pd.Series(self.filtered_stride[self.t_interp>startTime_auto,ss])
                autocorr_values = [signal.autocorr(lag=i) for i in range(len(signal)//2)]

                plot_time = 10
                # Subplot 1 (First row, spanning two columns)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, autocorr_values, linewidth=0.5)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, np.zeros(len(autocorr_values)),c='r', linewidth=2)
                #ax[ss].stem(range(len(autocorr_values)), autocorr_values,linefmt='b-', basefmt=" ", use_line_collection=True)
                ax[ss].set_title('Autocorrelation of ' + kp_list[ss])

                if ss==0:
                    # save autocorrelation value and lags in dataframe
                    autocorr_df = pd.DataFrame({'lags': np.arange(len(autocorr_values))/fps})
                autocorr_df[kp_list[ss]] = autocorr_values

            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis, 'Stride autocorrelation.png'), dpi=300)  # Save as PNG fil
            # save autocorrelation
            autocorr_df.to_csv(os.path.join(self.analysis, 'Stride autocorrelation.csv'))
            plt.close()

            #%%
            # instantaneous frequency with hilbert transform

            # Compute the analytic signal
            #
            # analytic_signal = hilbert(self.interp_stride[:,2])
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1 / fps))


            # Plot spectrogram

            # %% short time fourier transform
            # from scipy.signal import stft
            # frequencies, times, Zxx = stft(self.filtered_stride[:,3], fs=50, nperseg=256)
            # plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
            # plt.colorbar(label='Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [s]')
            # plt.title('STFT Magnitude')

            #%% pearson correlation between limbs
            # phase lag?
            # generate some plots
            pcorr = pd.DataFrame({'time': self.t_interp})
            corr_group = [['left hand','right hand'], ['left foot', 'right foot'],
                           ['left hand', 'left foot'], ['right hand', 'right foot']]
            corr_Idx = [[0,1], [2,3], [0,2], [1,3]]
            # xcorr between hands/feet/left/right
            timeStep = 2 # in second
            for kp_pairs,kp_idx in zip(corr_group,corr_Idx):
                corrCoeff_running = np.zeros((len(self.t_interp)))
                for idx,t in enumerate(self.t_interp):
                    tMask = np.logical_and(self.t_interp>t, self.t_interp <t+timeStep)
                    corrCoeff_running[idx] = np.corrcoef(self.filtered_stride[tMask,kp_idx[0]],
                                                             self.filtered_stride[tMask,kp_idx[1]])[0,1]
                pcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corrCoeff_running
            
            # cross correlation
            max_lag_sec = 1.0  # maximum lag to compute (in seconds)
            dt = self.t_interp[1] - self.t_interp[0]  # time step of your signal
            max_lag_samples = int(max_lag_sec / dt)

            # Store results
            max_xcorr = pd.DataFrame({'time': self.t_interp})
            max_lag = pd.DataFrame({'time': self.t_interp})

            for kp_pairs, kp_idx in zip(corr_group, corr_Idx):
                # Each element will be a 2D array: shape (len(t_interp), 2*max_lag_samples+1)
                # Arrays for max correlation and lag at each time point
                corr_max = np.full(len(self.t_interp), np.nan)
                lag_at_max = np.full(len(self.t_interp), np.nan)

                lags = np.arange(-max_lag_samples, max_lag_samples + 1) * dt

                for idx, t in enumerate(self.t_interp):
                    # 2-second window mask
                    tMask = (self.t_interp > t) & (self.t_interp < t + timeStep)
                    x = self.filtered_stride[tMask, kp_idx[0]]
                    y = self.filtered_stride[tMask, kp_idx[1]]

                    if len(x) < 2 or len(y) < 2:
                        continue

                    # Normalize signals
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Compute normalized cross-correlation
                    c = correlate(y, x, mode='full')
                    c = c / (np.std(x) * np.std(y) * len(x))

                    # Center index
                    mid = len(c) // 2
                    c_window = c[mid - max_lag_samples: mid + max_lag_samples + 1]

                    # Find max correlation and corresponding lag
                    max_idx = np.nanargmax(c_window)
                    corr_max[idx] = c_window[max_idx]
                    lag_at_max[idx] = lags[max_idx]
                
                max_xcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corr_max
                max_lag[kp_pairs[0]+'-'+kp_pairs[1]] = lag_at_max

            #save cross correlation results
            pcorr_renamed = pcorr.copy()
            pcorr_renamed.columns = ['time'] + [col + '_pearson' for col in pcorr.columns[1:]]

            max_xcorr_renamed = max_xcorr.copy()
            max_xcorr_renamed.columns = ['time'] + [col + '_maxxcorr' for col in max_xcorr.columns[1:]]

            max_lag_renamed = max_lag.copy()
            max_lag_renamed.columns = ['time'] + [col + '_lag' for col in max_lag.columns[1:]]

            # 2. Merge all DataFrames on 'time'
            combined_df = pcorr_renamed.merge(max_xcorr_renamed, on='time').merge(max_lag_renamed, on='time')

            # 3. Save to CSV
            combined_df.to_csv(os.path.join(self.analysis, 'Stride correlation.csv'), index=False)

            # make a plot to show pearson correlation and cross correlation and max lag
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-right hand'], label= 'Hands')
            ax[1].plot(self.t_interp, pcorr['left foot-right foot'], label = 'Feet')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-right hand'], label= 'Hands')
            ax[2].plot(self.t_interp, max_xcorr['left foot-right foot'], label = 'Feet')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-right hand'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['left foot-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - HF.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            # same plot to show left and right
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-left foot'], label= 'Left')
            ax[1].plot(self.t_interp, pcorr['right hand-right foot'], label = 'Right')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-left foot'], label= 'LEft')
            ax[2].plot(self.t_interp, max_xcorr['right hand-right foot'], label = 'Right')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-left foot'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['right hand-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - LR.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            #%% calculate hand/foot step amplitude and frequency based on peak detection
            self.stride_amp = []
            self.stride_time = []
            self.stride_freq = np.full(self.filtered_stride.shape, np.nan)

            time = self.t_interp
            for ll in range(4): # step size and amplitude of 4 limbs
            # Detect peaks (foot lifts)
            
                distance = self.filtered_stride[:,ll]
                peaks, props = find_peaks(distance, prominence=2, distance=None)

                # Estimate baseline before each step using local minima
                inv_distance = -distance
                troughs, _ = find_peaks(inv_distance, prominence=2 / 2, distance=None)

                step_amplitudes = []
                step_times = []

                for peak in peaks:
                    # Find the closest following trough (baseline)
                    next_troughs = troughs[troughs > peak]
                    if len(next_troughs) == 0:
                        continue
                    baseline_idx = next_troughs[0]
                    amplitude = distance[peak] - distance[baseline_idx]
                    step_amplitudes.append(amplitude)
                    step_times.append(time[peak])

                step_amplitudes = np.array(step_amplitudes)
                step_times = np.array(step_times)

                self.stride_amp.append(step_amplitudes)
                self.stride_time.append(step_times)


                # Compute step frequency (Hz) in running 2 second window
                window = 2
                freqs = np.full(len(time), np.nan)  # preallocate

                for i, t in enumerate(time):
                    # count steps within [t - window/2, t + window/2]
                    mask = (step_times >= t - window/2) & (step_times <= t + window/2)
                    steps_in_window = step_times[mask]

                    if len(steps_in_window) >= 1:
                        intervals = np.diff(steps_in_window)
                        freqs[i] = 1 / np.mean(intervals)
                    else:
                        freqs[i] = np.nan

                self.stride_freq[:,ll] = freqs


            fig,ax = plt.subplots(5, 1, figsize=(16, 16))
            # rod speed
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 2, stride of hand
            ax[1].plot(self.t_interp, self.filtered_stride[:,0])
            ax[1].plot(self.t_interp , self.filtered_stride[:,1])
            ax[1].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Distance between left/right hand and the rod')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 3 foot
            ax[2].plot(self.t_interp, self.filtered_stride[:,2])
            ax[2].plot(self.t_interp, self.filtered_stride[:,3])
            ax[2].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].set_title('Distance between left/right foot and the rod')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 hand amplitude
            ax[3].stem(self.stride_time[0], self.stride_amp[0], linefmt='C0-',  basefmt=" ", label='left hand')
            ax[3].stem(self.stride_time[1], self.stride_amp[1],linefmt='C1-',  basefmt=" ",label='right hand')
            ax[3].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[3].set_title('Hand step amplitude')
            ax[3].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 (Third row, first column)
            ax[4].stem(self.stride_time[2], self.stride_amp[2], linefmt='C0-',  basefmt=" ", label='left foot')
            ax[4].stem(self.stride_time[3], self.stride_amp[3], linefmt='C1-',  basefmt=" ", label='right foot')
            ax[4].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[4].set_title('Foot step amplitude')

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
            
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis,'Stride amplitude.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            data = {'left hand': self.filtered_stride[:,0],
                    'right hand': self.filtered_stride[:,1],
                    'left foot': self.filtered_stride[:, 2],
                    'right foot': self.filtered_stride[:, 3],
                    'stride amplitude': self.stride_amp,
                    'stride time': self.stride_time,
                    'stride frequency': self.stride_freq,
                    'time': self.t_interp}
            #dataDF = pd.DataFrame(data)
            #dataDF.to_csv(savedatapath)
            # save to pickle file
            with open( os.path.join(self.analysis, 'stride_freq.pickle'), 'wb') as f:
                pickle.dump(data, f)

            #%%
            # cumulative area under the curve
            # cum_xcorr_foot = np.cumsum(xcorr['left foot-right foot'])/fps
            # cum_xcorr_hand = np.cumsum(xcorr['left hand-right hand']) / fps
            # cum_xcorr_left = np.cumsum(xcorr['left hand-left foot'])/fps
            # cum_xcorr_right = np.cumsum(xcorr['right hand-right foot']) / fps
            # plt.figure()
            # plt.plot(self.t_interp, cum_xcorr_foot)
            # plt.plot(self.t_interp, cum_xcorr_hand)
            # plt.plot(self.t_interp, cum_xcorr_left)
            # plt.plot(self.t_interp, cum_xcorr_right)
            # plt.plot(self.data['rodT'][timeMaskRod], self.data['rodSpeed_smoothed'][timeMaskRod])
            # plt.xlabel('time')
            # plt.ylabel('Cumulative area under the curve of xcorr')
            # plt.legend(['feet','hands','left','right', 'Rod speed'])
            # plt.savefig(os.path.join(self.analysis,'Stride correlation.png'), dpi=300)  # Save as PNG fil
            # #plt.show()
            # plt.close()
            # # cross correlation in 10 second window

            # # save data in csv
            # xcorr.to_csv(os.path.join(self.analysis, 'Stride crosscorrelation.csv'))

            #
            #%% tail angle
            # calculate spine 3 - tail 1 - tail 2 angle
            A = np.array([self.data['spine 3']['x'], self.data['spine 3']['y']]).T
            B = np.array([self.data['tail 1']['x'], self.data['tail 1']['y']]).T
            C = np.array([self.data['tail 2']['x'], self.data['tail 2']['y']]).T

            A = A[timeMaskDLC,:]
            B = B[timeMaskDLC,:]
            C = C[timeMaskDLC,:]
            # Calculate vectors AB and BC
            AB = B - A
            BC = C - B

            # Calculate the angle between AB and BC
            # Calculate dot and cross products for each time point
            dot_product = np.sum(AB * BC, axis=1)  # Dot product for each row (time point)
            cross_product = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]  # Cross product for each time point

            # Calculate the angle at each time point
            angles = np.arctan2(cross_product, dot_product)

            # Convert to degrees
            angles = np.degrees(angles)

            # interpolate and filter the angle
            fps = 50
            self.filtered_angle = np.full((len(self.t_interp)), np.nan)
            self.interp_angle = np.full((len(self.t_interp)), np.nan)

            # need to determine the cutoff frequency here

                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
            self.interp_angle= np.interp(self.t_interp, time_include, angles)

            self.filtered_angle = butter_lowpass_filter(self.interp_angle, 5,fps,order=5)


            # save data in csv
            tail_angle= pd.DataFrame({'angle':self.filtered_angle, 'time':self.t_interp})
            tail_angle.to_csv(os.path.join(self.analysis, 'Tail angle.csv'))
            # Calculate the angle in radians using atan2 for correct sign

            # plot the video frame with keypoint estimatino
            # frame_num = 7760
            # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
            # plt.figure()
            # plt.imshow(curr_frame)
            # kp_plot = ['tail 2']
            # for kp in kp_plot:
            #     plt.scatter(self.data[kp]['x'][frame_num], self.data[kp]['y'][frame_num], s=20)

            #%% head angle

            #%% tail position
            # plot the density distribution of the tail
            # set coordinate of tail 1 to be (0, 0)
            # ego_tail = {}
            # tail_key = ['tail 1', 'tail 2', 'tail 3']
            # for t in tail_key:
            #     ego_tail[t] = {}
            #     ego_tail[t]['x']= np.array(self.data[t]['x'])-np.array(self.data['tail 1']['x'])
            #     ego_tail[t]['y'] = np.array(self.data[t]['y']) - np.array(self.data['tail 1']['y'])
            #
            # plt.figure(figsize=(12, 6))
            # # Density plot for aligned b coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 2']), x='x', y='y',
            #             fill=True, cmap='Blues', alpha=0.5, label='Point B',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm())
            # # Density plot for aligned c coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 3']), x='x', y='y',
            #             fill=True, cmap='Reds', alpha=0.5, label='Point C',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm()
            #             )
            # plt.axhline(0, color='black', lw=1, ls='--', label='y = 0')
            # plt.axvline(0, color='black', lw=1, ls='--', label='x = 0')
            #
            # plt.title('Density Distribution of Aligned Points B and C')
            # plt.xlabel('Aligned B X')
            # plt.ylabel('Aligned B Y / Aligned C Y')
            # plt.axhline(0, color='black', lw=0.5, ls='--')
            # plt.axvline(0, color='black', lw=0.5, ls='--')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # with open(savedatapath, 'wb') as f:
            #     pickle.dump(self.stride, self. f)
            # f.close()
        else:
            print("Analysis already done")
            return np.nan

    def stride_session(self):
        #todo: go over each trial and run get_stride for them
        pass

    def stride_summary(self):
                # things to do:
        # 1. foot amplitude and frequency in the beginning (5-20 rpm)

        #%% average cross correlation
        """ calculate the average cross correlation with in speed interval """
        startSpeed = np.arange(10,80,10)
        nTrials = 12
        stride = {}
        amp_std = {}
        amp_std_running = {}
        stride_amp_running = {}
        stride_freq_running = {}
        plot_speed = [] # keep the longest window speed to plot
        time_step = 5 # 2 s window
        # load stride frequency data
        keys = ['left hand', 'right hand', 'left foot', 'right foot']
        for key in keys:
            amp_std[key] = np.full((self.nSubjects, nTrials),np.nan)
            amp_std_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]
            stride_amp_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]
            stride_freq_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]

        amp_std['perf'] = np.full((self.nSubjects, nTrials),np.nan)
        genotype = self.GeneBG

        for idx, obj in enumerate(self.data['DLC_obj']):
            animal = self.data['Animal'][idx]
            trialIdx = self.data['Trial'][idx]-1
            animalIdx = self.animals.index(animal)

            if self.data['DLC'][idx] is not None:
                #%%  load the Stride_freq
                stridepickle = os.path.join(obj.analysis,'stride_freq.pickle')
                with open(stridepickle, 'rb') as handle:
                    stride = pickle.load(handle)
                rodSpeedCSV = os.path.join(obj.analysis, 'smoothed_rodSpeed.csv')
                rodSpeed = pd.read_csv(rodSpeedCSV)

                # isolate the time when animals turns around
                truncatedStride = copy.deepcopy(stride)
                bp_keys = ['left hand', 'right hand', 'left foot', 'right foot']
                for tInterval in obj.data['turning_period']:
                    tStart = max(stride['time'][0], obj.data['time'][tInterval[0]])
                    tEnd = min(stride['time'][len(stride['time'])-1],obj.data['time'][tInterval[1]])
                    nanMask = np.logical_and(stride['time']>=tStart, stride['time']<=tEnd)
                    for key in bp_keys:
                        truncatedStride[key][nanMask] = np.nan
                    
                    truncatedStride['stride frequency'][nanMask,:] = np.nan

                    # remove stride amplitude in the turning period
                    for sa in range(4):
                        nanMask_sa = np.logical_and(stride['stride time'][sa]>=tStart,
                                                    stride['stride time'][sa]<=tEnd)
                        truncatedStride['stride amplitude'][sa][nanMask_sa] = np.nan

                # calculate the average standard deviation of stride amplitude
                for kidx, key in enumerate(bp_keys):
                    amp_std[key][animalIdx, trialIdx] = np.nanstd(truncatedStride['stride amplitude'][kidx])
                
                if not self.data['FallByTurning'][np.logical_and(self.data['Animal']==animal,
                                                 self.data['Trial']==trialIdx+1)].any():
                    amp_std['perf'][animalIdx, trialIdx] = self.data['TimeOnRod'][np.logical_and(self.data['Animal']==animal,
                                                                                            self.data['Trial']==trialIdx+1)]
                
                # calculate running amplitude STD as a function of rod speed
                rod_time    = np.array(rodSpeed['time'])
                rod_speed   = np.array(rodSpeed['smoothed'])
                                    # --- Parameters ---
                window_size = 10.0    # 5 seconds window
                step_size   = 1    # sliding step 0.5 s

                # --- Determine start time: first time rod speed > 5 ---
                start_idx = np.where(rod_speed > 5)[0]
                if len(start_idx) == 0:
                    raise ValueError("Rod speed never exceeds 5")
                start_time = rod_time[start_idx[0]]

                # --- Generate sliding window time points ---
                below_zero_idx = np.where(rod_speed <= 0)[0]

                # Only consider "drops to zero" that happen *after* start_time
                below_zero_after_start = below_zero_idx[rod_time[below_zero_idx] > start_time]

                if len(below_zero_after_start) > 0:
                    # the first drop to zero after motion began
                    end_zero_time = rod_time[below_zero_after_start[0]]
                    end_time = end_zero_time - 2
                    # ensure it doesn’t go earlier than start_time
                    if end_time <= start_time:
                        end_time = rod_time[-1]
                else:
                    # if it never returns to zero
                    end_time = rod_time[-1]

                window_starts = np.arange(start_time, end_time - window_size + 0.01, step_size)
                
                for kidx, key in enumerate(bp_keys):
                    
                    stride_time = np.array(truncatedStride['stride time'][kidx])
                    stride_amp  = np.array(truncatedStride['stride amplitude'][kidx])


                    # --- Compute running amp, freq, and amp std for each window ---
                    running_std = []
                    running_amp = []
                    running_freq = []
                    window_centers = []

                    for t0 in window_starts:
                        t1 = t0 + window_size
                        mask = (stride_time >= t0) & (stride_time < t1)
                        if np.any(mask):
                            running_std.append(np.std(stride_amp[mask]))
                            running_amp.append(np.nanmean(stride_amp[mask]))
                            running_freq.append(np.sum(mask)/window_size)  # strides per second
                            window_centers.append(t0 + window_size/2)
                        else:
                            running_std.append(np.nan)  # no stride in window
                            running_amp.append(np.nan)
                            running_freq.append(np.nan)
                            window_centers.append(t0 + window_size/2)

                    # --- Convert to arrays for plotting ---
                    running_std = np.array(running_std)
                    running_amp = np.array(running_amp)
                    running_freq = np.array(running_freq)
                    window_centers = np.array(window_centers)
                    window_rod_speed = np.interp(window_centers, rod_time, rod_speed)
                    if len(window_rod_speed) > len(plot_speed):
                        plot_speed = window_rod_speed

                    amp_std_running[key][animalIdx][trialIdx]=running_std
                    stride_amp_running[key][animalIdx][trialIdx]=running_amp
                    stride_freq_running[key][animalIdx][trialIdx]=running_freq

                #%% load correlation 
                corrCSV = os.path.join(obj.analysis,'Stride correlation.csv')
                correlation = pd.read_csv(corrCSV)

                truncatedCorr = copy.deepcopy(correlation)
                corr_keys = correlation.keys().tolist()
                corr_keys.remove('time')
                for tInterval in obj.data['turning_period']:
                    tStart = max(stride['time'][0], obj.data['time'][tInterval[0]])
                    tEnd = min(stride['time'][len(stride['time'])-1],obj.data['time'][tInterval[1]])
                    nanMask = np.logical_and(correlation['time']>=tStart, correlation['time']<=tEnd)
                    for key in corr_keys:
                        truncatedCorr[key][nanMask] = np.nan

                # initialize variable in the beginning
                if idx == 0:
                    corr_summary = {}
                    for key in corr_keys:
                        corr_summary[key] =  [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)] 
                    corr_summary['rodSpeed'] =  [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)] 

                # take only time within start_time and end_time
                time_mask = np.logical_and(truncatedCorr['time']>=start_time, truncatedCorr['time']<=end_time)
                for key in corr_keys:
                    corr_summary[key][animalIdx][trialIdx] = truncatedCorr[key][time_mask]
                # interpolate time to rod speed
                corr_speed = np.interp(correlation['time'][time_mask], rod_time, rod_speed)
                corr_summary['rodSpeed'][animalIdx][trialIdx] = corr_speed
                
              



        #%% convert list to matrix, padding with NaN
        running_SD_matrix = {}
        for key in bp_keys: # convert running_std to matrix
                # find maximum length
            max_len = max(len(trial) for subj in amp_std_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(amp_std_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_SD_matrix[key] = data_3d

        running_amp_matrix = {}
        for key in bp_keys: # convert running_std to matrix
            # find maximum length
            max_len = max(len(trial) for subj in stride_amp_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(stride_amp_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_amp_matrix[key] = data_3d

        running_freq_matrix = {}
        for key in bp_keys: # convert running_std to matrix
            # find maximum length
            max_len = max(len(trial) for subj in stride_freq_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(stride_freq_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_freq_matrix[key] = data_3d

        corr_summary_matrix = {}
        for key in corr_keys:
            # Determine a common rod speed grid for interpolation
            # You can use the min/max across all trials to define it
            all_speeds = []
            for subj, subj_corr in zip(corr_summary[key], corr_summary['rodSpeed']):
                for trial, trial_speed in zip(subj, subj_corr):
                    if len(trial) > 0:
                        all_speeds.append(trial_speed)
            all_speeds = np.concatenate(all_speeds)
            
            # Define a common rod speed vector (e.g., 0.1 step)
            rod_speed_grid = np.arange(np.nanmin(all_speeds), np.nanmax(all_speeds)+0.1, 0.1)

            nSubjects = len(corr_summary[key])
            nTrials   = max(len(subj) for subj in corr_summary[key])
            nSpeeds   = len(rod_speed_grid)

            # Initialize 3D matrix with NaNs
            data_3d = np.full((nSubjects, nTrials, nSpeeds), np.nan)

            for i, subj_corr in enumerate(corr_summary[key]):
                for j, trial_corr in enumerate(subj_corr):
                    trial_speed = corr_summary['rodSpeed'][i][j]  # corresponding rod speed for this trial
                    if len(trial_corr) > 0:
                        # Interpolate trial data onto common rod speed grid
                        data_interp = np.interp(
                            rod_speed_grid,         # new x (grid)
                            trial_speed,            # original x
                            trial_corr,             # original y
                            left=np.nan,            # pad out-of-bounds with NaN
                            right=np.nan
                        )
                        data_3d[i, j, :] = data_interp

            corr_summary_matrix[key] = (rod_speed_grid, data_3d)
        
 
        #%% plot performance - stride std correlation 
        for key in bp_keys:
            perf = np.array(amp_std['perf'])          # shape (15, 12)
            left_sd = np.array(amp_std[key])  # shape (15, 12)
            genotype = np.array(genotype)             # length 15, entries 'WT' or 'KO'

            # --- masks ---
            wt_mask = genotype == 'WT'
            ko_mask = genotype == 'KO'

            # --- flatten + remove NaN for correlation ---
            def clean_flatten(mask):
                x = perf[mask].flatten()
                y = left_sd[mask].flatten()
                valid = ~np.isnan(x) & ~np.isnan(y)
                return x[valid], y[valid]

            perf_wt, sd_wt = clean_flatten(wt_mask)
            perf_ko, sd_ko = clean_flatten(ko_mask)

            # --- Pearson correlation ---
            r_wt, p_wt = pearsonr(perf_wt, sd_wt)
            r_ko, p_ko = pearsonr(perf_ko, sd_ko)

            # --- plotting setup ---
            trials = np.arange(12)
            norm = Normalize(vmin=0, vmax=11)
            cmap_wt = plt.cm.Greys
            cmap_ko = plt.cm.Reds

            plt.figure(figsize=(7,6))

            # --- scatter points ---
            for i in range(len(genotype)):
                cmap = cmap_wt if genotype[i] == 'WT' else cmap_ko
                colors = cmap(norm(trials))
                for t in trials:
                    x, y = perf[i, t], left_sd[i, t]
                    if not np.isnan(x) and not np.isnan(y):
                        plt.scatter(y, x, color=colors[t], s=60, edgecolor='none')

            # --- correlation text ---
            plt.text(0.05, 0.95, f"WT: r={r_wt:.2f}, p={p_wt:.3g}", transform=plt.gca().transAxes,
                    color='black', fontsize=10, va='top')
            plt.text(0.05, 0.88, f"KO: r={r_ko:.2f}, p={p_ko:.3g}", transform=plt.gca().transAxes,
                    color='red', fontsize=10, va='top')

            # --- legend 1: genotype ---
            genotype_handles = [
                Patch(facecolor='black', label='WT'),
                Patch(facecolor='red', label='KO')
            ]
            legend1 = plt.legend(handles=genotype_handles, loc='upper right', frameon=False)

            # --- legend 2: trial gradient ---
            sm = ScalarMappable(norm=norm, cmap=cmap_ko)
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
            cbar.set_label('Trial #', rotation=270, labelpad=15)
            cbar.set_ticks([0, 3, 6, 9])
            cbar.set_ticklabels(['1', '4', '7', '10'])

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # --- labels ---
            plt.ylabel('Performance')
            plt.xlabel('Amplitude SD')
            plt.title('Performance vs ' + key + ' Amplitude SD')

            plt.gca().add_artist(legend1)
            plt.tight_layout()
            plt.show()

            # save fig in png and svg format
            savefigpath = os.path.join(self.sumFolder, 'Performance vs ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Performance vs ' + key + ' Amplitude SD.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot running std vs rod speed for different gnotype

        # mixed ANOVA 
        for key in bp_keys:

            # Example dimensions
            data_3d = running_SD_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax1.set_title('Stride variability ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Stride variability ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + ' Amplitude SD.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average frequency and amplitude vs rod speed
        for key in bp_keys:

            # Example dimensions
            data_3d = running_amp_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride amplitude (mean ± STE)')
            ax1.set_title('Stride amplitude ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Average stride amplitude ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Average Amplitude.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + 'Average Amplitude.svg')
            plt.savefig(savefigpath, format='svg')

        for key in bp_keys:

            # Example dimensions
            data_3d = running_freq_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride frequency (mean ± STE)')
            ax1.set_title('Stride frequency ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Average stride frequency ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Average Frequency.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + 'Average Frequency.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average correlation vs rod speed for different genotype
        for key in corr_keys:

            # Example dimensions
            data_3d = corr_summary_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d[1].shape
            plot_speed = data_3d[0] # x axis
            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d[1], axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'dependentVar': data_3d[1][i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['dependentVar', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("dependentVar ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues
            print(result.summary())
            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['dependentVar']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel(key+' (mean ± STE)')
            ax1.set_title(key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['dependentVar']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title(key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + '.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + '.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average amplitude/frequency at 5-20 RPM within trial 1-3, 4-6, 7-9, and 10-12
        
class DLCSession:

    def __init__(self, filePath, videoPath, rodspeedPath,analysisPath, fps):
        """

        :param filePath: DLC csv path
        :param videoPath:
        :param rodspeedPath: rod speed csv path
        :param analysisPath:
        :param fps: number (frames per second); path: file path with time stamps
        """

        self.filePath = filePath
        self.videoPath = videoPath
        self.rodPath = rodspeedPath
        self.nFrames = 0
        if fps.isnumeric():
            self.fps = fps
        else:
            # load the timeStamp csv
            time_raw = pd.read_csv(fps, header=None)
            self.t = np.array(time_raw[0]-time_raw[0][0])/1000
            self.t_start = time_raw[0][0]
        # read data
        self.data = self.read_data()
        self.analysis = analysisPath
        self.fieldSize = 40 # in centimeter, used to convert px to cm
        if not os.path.exists(self.analysis):
            os.makedirs(self.analysis)
        #self.video = self.read_video()

    def get_confidence(self, p_threshold, savefigpath):
        # get potentially outlier frames by confidence
        # focus on body parts that are most likely to be wrong:
        # tail 1, nose, left/right hand/foot
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'spine 1', 'spine 2', 'spine 3', 'tail 1']
        outliers = []
        for f in range(self.nFrames):
            for bp in body_parts:
                if self.data[bp]['p'][f]<p_threshold:
                    outliers.append(f)
                break

        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
        #self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath,figName))
            plt.close()

    def get_jump(self, px_threshold, savefigpath):
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'tail 1', 'tail 2', 'tail 3']
        outliers = []
        for f in range(self.nFrames-1):
            for bp in body_parts:
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                if np.sqrt(dx2+dy2) > px_threshold:
                    outliers.append(f+1)
                break


        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
            # self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath, figName))
            plt.close()

    def kp_jump_dist(self):
        # calculate cross frame keypoint jumps and plot the distrubution
        body_parts = self.data['bodyparts']
        kp_jumps = {}
        for f in range(self.nFrames-1):
            for bp in body_parts:
                if f==0:
                    kp_jumps[bp] = []
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                kp_jumps[bp].append(np.sqrt(dx2+dy2))

        bins = np.arange(0.0,30,0.2)
        fig, ax = plt.subplots(3,5,sharey=True)
        for idx, bp in enumerate(body_parts):
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].hist(kp_jumps[bp], bins= bins)
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].set_xlabel(bp)

    def moving_trace(self, savefigpath):
        """ plot animal moving trace in the field"""
        if not hasattr(self, 'arena'):
            savedatapath = os.path.join(savefigpath, 'arena_coordinates.csv')
            if not os.path.exists(savedatapath):
                self.arena = frame_input(self.videoPath)
                # save the results:
                with open(savedatapath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['upper left',
                                     'upper right',
                                     'lower right',
                                     'lower left'])
                    writer.writerow([self.arena['upper left'],
                                    self.arena['upper right'],
                                    self.arena['lower right'],
                                    self.arena['lower left']])
                    f.close()
            else:
                # read data from file
                tempdata= pd.read_csv(savedatapath)
                self.arena = {}
                for key in tempdata.keys():
                    self.arena[key] = ast.literal_eval(tempdata[key].values[0])

                # convert px to cm
                # calculate the length of each side in pixels, get the average, then convert to cm
        sideLength = []
        arenaKeys = list(self.arena.keys())
        for kidx in range(len(arenaKeys)):
            key1 = arenaKeys[kidx]
            if kidx < len(arenaKeys)-1:
                key2 = arenaKeys[kidx + 1]
            else:
                key2 = arenaKeys[0]
            sideLength.append(np.sqrt((self.arena[key1][0]-self.arena[key2][0])**2+
                                              (self.arena[key1][1]-self.arena[key2][1])**2))
        # save the coordinates in analysis folder
        self.px2cm = self.fieldSize/np.mean(sideLength)

        arena_x = [self.arena['upper left'][0], self.arena['upper right'][0],
                   self.arena['lower right'][0], self.arena['lower left'][0], self.arena['upper left'][0]]
        arena_y = [self.arena['upper left'][1], self.arena['upper right'][1],
                   self.arena['lower right'][1], self.arena['lower left'][1], self.arena['upper left'][1]]

        # get the instantaneous distance from center
        slope1 = (self.arena['upper left'][1] - self.arena['lower right'][1]) / (self.arena['upper left'][0] - self.arena['lower right'][0])
        slope2 = (self.arena['lower left'][1] - self.arena['upper right'][1]) / (self.arena['lower left'][0] - self.arena['upper right'][0])

        # Calculate the x-coordinate of the intersection point
        x_intersection = ((self.arena['upper right'][1] - self.arena['upper left'][1]) + slope1 * self.arena['upper left'][0] - slope2 * self.arena['upper right'][0]) / (slope1 - slope2)

        # Calculate the y-coordinate of the intersection point
        y_intersection = slope1 * (x_intersection - self.arena['upper left'][0]) + self.arena['upper left'][1]

        self.center_point = [x_intersection, y_intersection]

        self.dist_center = np.sqrt((np.array(self.data['tail 1']['x'])-self.center_point[0])**2 +
                                   (np.array(self.data['tail 1']['y'])-self.center_point[1])**2)

        if hasattr(self, 'px2cm'):
            self.dist_center = self.dist_center*self.px2cm

        tracePlot = StartPlots()
        tracePlot.ax.plot(arena_x, arena_y)
        tracePlot.ax.plot(self.data['tail 1']['x'], self.data['tail 1']['y'])
        tracePlot.ax.axis('equal')
        # Hide the x and y axes
        tracePlot.ax.axis('off')
        tracePlot.save_plot('Moving trace.tif', 'tif', savefigpath)
        tracePlot.save_plot('Moving trace.svg', 'svg', savefigpath)

    def get_time_in_center(self):
        """calculate time spent in the center"""
        if hasattr(self, 'arena'):
            # do the calculation
            side_length = np.sqrt((self.arena['upper left'][0] - self.arena['upper right'][0])**2 + (self.arena['upper left'][1] - self.arena['upper right'][1])**2)
            self.center = {}
            self.center['upper left'] = (self.arena['upper left'][0] + side_length/4,
                                         self.arena['upper left'][1] + side_length/4)
            self.center['upper right'] = (self.arena['upper right'][0] - side_length/4,
                                          self.arena['upper right'][1] + side_length / 4)
            self.center['lower right'] = (self.arena['lower right'][0] - side_length/4,
                                             self.arena['lower right'][1] - side_length / 4)
            self.center['lower left'] = (self.arena['lower left'][0] + side_length/4,
                                            self.arena['lower right'][1] - side_length / 4)

            # determine if tail 1 is inthe center area\
            x_left = (self.center['upper left'][0] + self.center['lower left'][0])/2
            x_right = (self.center['upper right'][0] + self.center['lower right'][0])/2
            y_upper = (self.center['upper left'][1] + self.center['upper right'][1])/2
            y_lower = (self.center['lower left'][1] + self.center['lower right'][1])/2

            is_center = np.zeros(len(self.data['tail 1']['x']))
            num_cross = 0
            self.num_cross = []  # number of times the animal crosses the border line of center area
            for idx in range(len(self.data['tail 1']['x'])):
                if self.data['tail 1']['x'][idx] > x_left and self.data['tail 1']['x'][idx] < x_right:
                    if self.data['tail 1']['y'][idx] > y_upper and self.data['tail 1']['y'][idx] < y_lower:
                        is_center[idx] = 1

                        if idx > 0:
                            if is_center[idx] != is_center[idx-1]:
                                num_cross+=1
                self.num_cross.append(num_cross)

            self.time_in_center = is_center
            self.cumu_time_center = []
            cumu = 0
            for f in range(self.nFrames):
                cumu += self.time_in_center[f]/self.fps
                self.cumu_time_center.append(cumu)

        else:
            print("please run moving_trace first")

    def plot_distance_to_center(self, t, savefigpath):
        # plot the distribution of distance to center
        # as well as a function of time
        distPlot = StartPlots()
        self.dist_center_bins = distPlot.ax.hist(self.dist_center, bins = np.linspace(0, 1200, 101))
        self.dist_center_bins_30 = distPlot.ax.hist(self.dist_center[0:30*60*self.fps],
                                                    bins = np.linspace(0,1200, 101))
        distPlot.ax.set_xlabel('Distance from center (px)')
        distPlot.ax.set_ylabel('Occurance')
        distPlot.save_plot('Distribution of distance from center.tiff', 'tiff', savefigpath)
        # average distance from center in a running window
        self.dist_center_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
        for ff in range(self.nFrames - 1 - t*self.fps):
            self.dist_center_running[ff] = np.nanmean(self.dist_center[ff:ff+t*self.fps])

        distRunningPlot = StartPlots()
        distRunningPlot.ax.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.dist_center_running)
        distRunningPlot.ax.set_ylabel('Average distance from center (px)')
        distRunningPlot.ax.set_xlabel('Time (s)')

        distRunningPlot.save_plot('Average distance from center.tiff', 'tiff', savefigpath)
        plt.close('all')

    def read_data(self):
        data = {}
        if not hasattr(self, 't'):
            self.t = []

        if isinstance(self.filePath, str):
            with open(self.filePath) as csv_file:
                print("Loading data from: " + self.filePath)
                csv_reader = csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:  # scorer
                        data[row[0]] = row[1]
                        line_count += 1
                    elif line_count == 1:  # body parts
                        bodyPartList = []
                        for bb in range(len(row) - 1):
                            if row[bb + 1] not in bodyPartList:
                                bodyPartList.append(row[bb + 1])
                        data[row[0]] = bodyPartList
                        #print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    elif line_count == 2:  # coords
                        #print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    elif line_count == 3:  # actual coords
                        # print({", ".join(row)})
                        tempList = ['x', 'y', 'p']
                        for ii in range(len(row) - 1):
                            # get the corresponding body parts based on index
                            body = data['bodyparts'][int(np.floor((ii) / 3))]
                            if np.mod(ii, 3) == 0:
                                data[body] = {}
                            data[body][tempList[np.mod(ii, 3)]] = [float(row[ii + 1])]
                        #self.t.append(0)
                        line_count += 1
                        self.nFrames += 1

                    else:
                        tempList = ['x', 'y', 'p']
                        for ii in range(len(row) - 1):
                            # get the corresponding body parts based on index
                            body = data['bodyparts'][int(np.floor((ii) / 3))]
                            data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                        #self.t.append(self.nFrames*(1/self.fps))
                        line_count += 1
                        self.nFrames += 1

                print(f'Processed {line_count} lines.')

                # add frame time
                #tStep= 1/self.fps
                data['time'] = self.t
                #self.t = np.array(self.t)
        else:
            data['time'] = np.nan
        # load rod speed data
        rodSpeed = pd.read_csv(self.rodPath, header=None)
        data['rodSpeed'] = rodSpeed.iloc[:, 0].values
        data['rodT'] = (rodSpeed.iloc[:, 1].values-self.t_start)/1000

        #%% for estimations with likelihood less than 0.8, replace the value with linear fit
        # based on previous and next value
        # corrected_data=copy.deepcopy(data)
        # kp_list = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot',
        #            'nose', 'left ear', 'right ear','left hand', 'right hand']
        # corrected_frames = []
        # for kp in kp_list:
        #     data[kp]['x'] = np.array(data[kp]['x'])
        #     data[kp]['y'] = np.array(data[kp]['y'])
        #     data[kp]['p'] = np.array(data[kp]['p'])
        #     for i in range(6, len(data['time'])-6):
        #         if data[kp]['p'][i] < 0.8:
        #             prev_reliable={}
        #             next_reliable = {}
        #             prev_reliable['x'] = data[kp]['x'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             prev_reliable['y'] = data[kp]['y'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             next_reliable['x'] = data[kp]['x'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             next_reliable['y'] = data[kp]['y'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             prev_reliable = pd.DataFrame(prev_reliable)
        #             next_reliable = pd.DataFrame(next_reliable)
        #             reliable_points = pd.concat([prev_reliable, next_reliable])
        #
        #             # If we found any reliable points, replace the unreliable x and y values
        #             if not reliable_points.empty:
        #                 if not i in corrected_frames:
        #                     corrected_frames.append(i)
        #                 # Calculate average x and y of these reliable points
        #                 avg_x = reliable_points['x'].mean()
        #                 avg_y = reliable_points['y'].mean()
        #
        #                 # Replace unreliable x and y values with the interpolated average
        #                 corrected_data[kp]['x'][i] = avg_x
        #                 corrected_data[kp]['y'][i] = avg_y
        #
        # # plot some frames to examine it
        # frame_num = 10198
        # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
        # plt.figure()
        # plt.imshow(curr_frame)
        # cmap = cm.get_cmap('viridis', len(kp_list))
        # for kp in kp_list:
        #     plt.scatter(data[kp]['x'][frame_num], data[kp]['y'][frame_num], c=cmap(kp_list.index(kp)), s=200,label = kp)
        #     plt.scatter(corrected_data[kp]['x'][frame_num],corrected_data[kp]['y'][frame_num],marker = 'x',c=cmap(kp_list.index(kp)), s=200,label = kp+'_corrected')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return data

    def check_quality(self):
        # go through the data and plot the distribution of p-values
        pass

    def get_movement(self):
        # calculate distance, running velocity, acceleration, based on tail (base of tail)
        savedatapath = os.path.join(self.analysis,'movement.pickle')
        if not os.path.exists(savedatapath):
            self.vel = np.zeros((self.nFrames-1, 1))
            self.dist = np.zeros((self.nFrames-1,1))
            self.accel = np.zeros((self.nFrames-1, 1))

            for ff in range(self.nFrames-1):
                self.dist[ff] = np.sqrt((self.data['tail 1']['x'][ff+1] - self.data['tail 1']['x'][ff])**2 +
                    (self.data['tail 1']['y'][ff + 1] - self.data['tail 1']['y'][ff]) ** 2)

                self.vel[ff] = (self.dist[ff])*self.fps
                if ff<self.nFrames-2:
                    self.accel[ff] = (self.vel[ff+1]-self.vel[ff])*self.fps
            # save vel, dist, accel in pickle file
            dist = self.dist
            vel = self.vel
            accel = self.accel
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist, vel, accel], f)
            f.close()
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist, self.vel, self.accel = pickle.load(f)
            f.close()

        if hasattr(self, 'px2cm'):
            self.dist = self.dist*self.px2cm
            self.vel = self.vel*self.px2cm
            self.accel = self.accel*self.px2cm

    def get_movement_running(self, t, savefigpath):
        savedatapath = os.path.join(self.analysis, 'movement_running.pickle')
        if not os.path.exists(savedatapath):
        # get average distance and velocity in running window of t seconds
            self.vel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.dist_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.accel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.dist_running[ff] = np.sum(self.dist[ff:ff+t*self.fps])
                self.vel_running[ff] = np.nanmean(self.vel[ff:ff+t*self.fps])
                self.accel_running[ff] = np.nanmean(self.accel[ff:ff+t*self.fps])
                self.vel[ff] = (self.dist[ff]) * self.fps

            dist_running = self.dist_running
            vel_running = self.vel_running
            accel_running = self.accel_running
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist_running, vel_running, accel_running], f)
            f.close()

            velPlot = StartPlots()
            velPlot.ax.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.dist_running)
            velPlot.ax.set_ylabel('Average distance traveled (px)')
            #ax2 = velPlot.ax.twinx()
            #ax2.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.vel_running, color='red')
            #ax2.set_ylabel('Average velocity')
            velPlot.ax.set_xlabel('Time (s)')

            velPlot.save_plot('Running distance and velocity.png', 'png', savefigpath)
            plt.close(velPlot.fig)
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist_running, self.vel_running, self.accel_running = pickle.load(f)
            f.close()
        if hasattr(self, 'px2cm'):
            self.dist_running = self.dist_running*self.px2cm
            self.vel_running= self.vel_running*self.px2cm
            self.accel_running = self.accel_running*self.px2cm

    def get_angular_velocity(self):
        # calculate angular velocity based on tail and spine 1
        savedatapath = os.path.join(self.analysis, 'angular_velocity.pickle')
        if not os.path.exists(savedatapath):
            self.angVel = np.zeros((self.nFrames-1, 1))
            for ff in range(self.nFrames-1):
                y1 = self.data['spine 1']['y'][ff] - self.data['tail 1']['y'][ff]
                x1 = self.data['spine 1']['x'][ff] - self.data['tail 1']['x'][ff]

                y2 = self.data['spine 1']['y'][ff+1] - self.data['tail 1']['y'][ff+1]
                x2 = self.data['spine 1']['x'][ff+1] - self.data['tail 1']['x'][ff+1]

                self.angVel[ff] = self.get_angle([x1, y1], [x2, y2])*self.fps
            angVel = self.angVel
            with open(savedatapath, 'wb') as f:
                pickle.dump(angVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel = pickle.load(f)
            f.close()
        #self.angVel = self.angVel*self.fps

    def get_head_angular_velocity(self):
        savedatapath = os.path.join(self.analysis, 'head_angular_velocity.pickle')
        if not os.path.exists(savedatapath):
            self.headAngVel = np.zeros((self.nFrames, 1))
            for ff in range(self.nFrames-1):
                # get the mid point of two ears
                midX1 = (self.data['left ear']['x'][ff] + self.data['right ear']['x'][ff])/2
                midY1 = (self.data['left ear']['y'][ff] + self.data['right ear']['y'][ff])/2

                midX2 = (self.data['left ear']['x'][ff+1] + self.data['right ear']['x'][ff+1])/2
                midY2 = (self.data['left ear']['y'][ff+1] + self.data['right ear']['y'][ff+1])/2

                v1 = [self.data['nose']['x'][ff]-midX1, self.data['nose']['y'][ff]-midY1]
                v2 = [self.data['nose']['x'][ff+1]-midX2, self.data['nose']['y'][ff+1]-midY2]

                self.headAngVel[ff] = self.get_angle(v1, v2) * self.fps
            with open(savedatapath, 'wb') as f:
                pickle.dump(self.headAngVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.headAngVel = pickle.load(f)
            f.close()

    def get_stride(self,front_kp, back_kp, df_entry):
        savedatapath = os.path.join(self.analysis, 'stride_freq.csv')
        runFile = os.path.join(self.analysis, 'notExist.csv') # a not existing file to allow re-calculating
        if not os.path.exists(runFile):

            savefigpath = os.path.join(self.analysis)
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)

            # get rid the the turning period
            
            # %% define rod plane first
            # load reference point
            ave_left_rod_back = self.data['left_rod_back']
            ave_right_rod_back = self.data['right_rod_back']
            ave_center_rod_back = self.data['center_rod_back']
            ave_left_rod_front = self.data['left_rod_front']
            ave_right_rod_front = self.data['right_rod_front']
            ave_center_rod_front = self.data['center_rod_front']

            ref_plot = os.path.join(savefigpath, 'Rod coordinate.png')
            if not os.path.exists(ref_plot):
                frame = read_video(self.videoPath, 0, ifgray=False)
                # overlay the video frame?
                plt.figure()
                plt.imshow(frame)
                plt.scatter(self.data['rod_left_back']['x'], self.data['rod_left_back']['y'],
                            c=self.data['rod_left_back']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood
                plt.colorbar(label='Confidence')

                plt.scatter(self.data['rod_right_back']['x'], self.data['rod_right_back']['y'],
                            c=self.data['rod_right_back']['p'], cmap='viridis', s=100)

                plt.scatter(self.data['rod_left_front']['x'], self.data['rod_left_front']['y'],
                            c=self.data['rod_left_front']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood

                plt.scatter(self.data['rod_right_front']['x'], self.data['rod_right_front']['y'],
                            c=self.data['rod_right_front']['p'], cmap='viridis', s=100)

                # get average from keypoints with confidence higher than 95


                plt.scatter(ave_left_rod_back[0],ave_left_rod_back[1], s=500)
                plt.scatter(ave_right_rod_back[0], ave_right_rod_back[1], s=500)
                plt.scatter(ave_center_rod_back[0], ave_center_rod_back[1], s=500)

                plt.scatter(ave_left_rod_front[0],ave_left_rod_front[1], s=500)
                plt.scatter(ave_right_rod_front[0], ave_right_rod_front[1], s=500)
                plt.scatter(ave_center_rod_front[0], ave_center_rod_front[1], s=500)

                plt.savefig(os.path.join(savefigpath, 'rod_plane.png'))
                plt.close()
            # save the figure

            # %% examine the body parts in back view and front view

            # find behavior time (from rod start turning to fall)
            startTime= self.data['rodT'][self.data['rodSpeed_smoothed']>0][0]
            if np.isnan(self.data['rodStart'][0]):
                self.data['rodStart'][0] = 0
            endTime = startTime+df_entry['TimeOnRod'] + self.data['rodRun'][0] - self.data['rodStart'][0] # need the time stay on rod

            timeMaskDLC = np.logical_and(self.data['time']>=startTime, self.data['time']<= endTime)
            timeMaskRod = np.logical_and(self.data['rodT']>=startTime, self.data['rodT']<= endTime)
            nFramesInclude = np.sum(timeMaskDLC)
            time_include = self.data['time'][timeMaskDLC]
            kp_list = ['left hand', 'right hand', 'left foot', 'right foot']
            self.stride = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_front[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_front[1])**2)
                elif 'foot' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_back[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_back[1])**2)

            # try calculate the stride using distance from the rod (a horizontal line)
            self.stride_rod = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_left_rod_front, ave_right_rod_front)
                elif 'foot' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_right_rod_back, ave_left_rod_back)

            #tempMask = ~p_mask[timeMask]
            #    self.stride[tempMask,idx] = np.nan
            #    tempStride, tempIdx = fill_nans_and_split(self.stride[:, idx])
                # interpolate the nans
                # self.notnanChunks[kp] = tempIdx
                # for ich, chunk in enumerate(tempIdx):
                #     self.stride[chunk[0]:chunk[1]+1,idx] = tempStride[ich]


            # low-pass filter the data
            fps = 50
            self.t_interp = np.arange(time_include[0], time_include[-1] + 1 / fps, 1 / fps)
            self.filtered_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)
            self.interp_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)

            # need to determine the cutoff frequency here
            for idx, kp in enumerate(kp_list):
                #for ich, chunk in enumerate(self.notnanChunks[kp]):
                #    if chunk[1]-chunk[0]+1 > 18:  # padlen
                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
                self.interp_stride[:,idx] = np.interp(self.t_interp, time_include, self.stride_rod[:,idx])

                self.filtered_stride[:,idx] = butter_lowpass_filter(self.interp_stride[:,idx], 5,fps,order=5)

            #%%
            # examine the autocorrelation
            # average them over genotype and trial
            # find the time when rod speed reach 5/10
            #if df_entry['Trial']<=6:
            startSpeed = 5
            #else:
            #    startSpeed = 10
            startTime_auto = self.data['rodT'][self.data['rodSpeed_smoothed']>startSpeed][0]
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 subplots
            ax = ax.flatten()
            for ss in range(len(kp_list)):
                signal = pd.Series(self.filtered_stride[self.t_interp>startTime_auto,ss])
                autocorr_values = [signal.autocorr(lag=i) for i in range(len(signal)//2)]

                plot_time = 10
                # Subplot 1 (First row, spanning two columns)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, autocorr_values, linewidth=0.5)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, np.zeros(len(autocorr_values)),c='r', linewidth=2)
                #ax[ss].stem(range(len(autocorr_values)), autocorr_values,linefmt='b-', basefmt=" ", use_line_collection=True)
                ax[ss].set_title('Autocorrelation of ' + kp_list[ss])

                if ss==0:
                    # save autocorrelation value and lags in dataframe
                    autocorr_df = pd.DataFrame({'lags': np.arange(len(autocorr_values))/fps})
                autocorr_df[kp_list[ss]] = autocorr_values

            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis, 'Stride autocorrelation.png'), dpi=300)  # Save as PNG fil
            # save autocorrelation
            autocorr_df.to_csv(os.path.join(self.analysis, 'Stride autocorrelation.csv'))
            plt.close()

            #%%
            # instantaneous frequency with hilbert transform

            # Compute the analytic signal
            #
            # analytic_signal = hilbert(self.interp_stride[:,2])
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1 / fps))


            # Plot spectrogram

            # %% short time fourier transform
            # from scipy.signal import stft
            # frequencies, times, Zxx = stft(self.filtered_stride[:,3], fs=50, nperseg=256)
            # plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
            # plt.colorbar(label='Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [s]')
            # plt.title('STFT Magnitude')

            #%% pearson correlation between limbs
            # phase lag?
            # generate some plots
            pcorr = pd.DataFrame({'time': self.t_interp})
            corr_group = [['left hand','right hand'], ['left foot', 'right foot'],
                           ['left hand', 'left foot'], ['right hand', 'right foot']]
            corr_Idx = [[0,1], [2,3], [0,2], [1,3]]
            # xcorr between hands/feet/left/right
            timeStep = 2 # in second
            for kp_pairs,kp_idx in zip(corr_group,corr_Idx):
                corrCoeff_running = np.zeros((len(self.t_interp)))
                for idx,t in enumerate(self.t_interp):
                    tMask = np.logical_and(self.t_interp>t, self.t_interp <t+timeStep)
                    corrCoeff_running[idx] = np.corrcoef(self.filtered_stride[tMask,kp_idx[0]],
                                                             self.filtered_stride[tMask,kp_idx[1]])[0,1]
                pcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corrCoeff_running
            
            # cross correlation
            max_lag_sec = 1.0  # maximum lag to compute (in seconds)
            dt = self.t_interp[1] - self.t_interp[0]  # time step of your signal
            max_lag_samples = int(max_lag_sec / dt)

            # Store results
            max_xcorr = pd.DataFrame({'time': self.t_interp})
            max_lag = pd.DataFrame({'time': self.t_interp})

            for kp_pairs, kp_idx in zip(corr_group, corr_Idx):
                # Each element will be a 2D array: shape (len(t_interp), 2*max_lag_samples+1)
                # Arrays for max correlation and lag at each time point
                corr_max = np.full(len(self.t_interp), np.nan)
                lag_at_max = np.full(len(self.t_interp), np.nan)

                lags = np.arange(-max_lag_samples, max_lag_samples + 1) * dt

                for idx, t in enumerate(self.t_interp):
                    # 2-second window mask
                    tMask = (self.t_interp > t) & (self.t_interp < t + timeStep)
                    x = self.filtered_stride[tMask, kp_idx[0]]
                    y = self.filtered_stride[tMask, kp_idx[1]]

                    if len(x) < 2 or len(y) < 2:
                        continue

                    # Normalize signals
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Compute normalized cross-correlation
                    c = correlate(y, x, mode='full')
                    c = c / (np.std(x) * np.std(y) * len(x))

                    # Center index
                    mid = len(c) // 2
                    c_window = c[mid - max_lag_samples: mid + max_lag_samples + 1]

                    # Find max correlation and corresponding lag
                    max_idx = np.nanargmax(c_window)
                    corr_max[idx] = c_window[max_idx]
                    lag_at_max[idx] = lags[max_idx]
                
                max_xcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corr_max
                max_lag[kp_pairs[0]+'-'+kp_pairs[1]] = lag_at_max

            #save cross correlation results
            pcorr_renamed = pcorr.copy()
            pcorr_renamed.columns = ['time'] + [col + '_pearson' for col in pcorr.columns[1:]]

            max_xcorr_renamed = max_xcorr.copy()
            max_xcorr_renamed.columns = ['time'] + [col + '_maxxcorr' for col in max_xcorr.columns[1:]]

            max_lag_renamed = max_lag.copy()
            max_lag_renamed.columns = ['time'] + [col + '_lag' for col in max_lag.columns[1:]]

            # 2. Merge all DataFrames on 'time'
            combined_df = pcorr_renamed.merge(max_xcorr_renamed, on='time').merge(max_lag_renamed, on='time')

            # 3. Save to CSV
            combined_df.to_csv(os.path.join(self.analysis, 'Stride correlation.csv'), index=False)

            # make a plot to show pearson correlation and cross correlation and max lag
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-right hand'], label= 'Hands')
            ax[1].plot(self.t_interp, pcorr['left foot-right foot'], label = 'Feet')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-right hand'], label= 'Hands')
            ax[2].plot(self.t_interp, max_xcorr['left foot-right foot'], label = 'Feet')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-right hand'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['left foot-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - HF.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            # same plot to show left and right
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-left foot'], label= 'Left')
            ax[1].plot(self.t_interp, pcorr['right hand-right foot'], label = 'Right')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-left foot'], label= 'LEft')
            ax[2].plot(self.t_interp, max_xcorr['right hand-right foot'], label = 'Right')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-left foot'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['right hand-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - LR.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            #%% calculate hand/foot step amplitude and frequency based on peak detection
            self.stride_amp = []
            self.stride_time = []
            self.stride_freq = np.full(self.filtered_stride.shape, np.nan)

            time = self.t_interp
            for ll in range(4): # step size and amplitude of 4 limbs
            # Detect peaks (foot lifts)
            
                distance = self.filtered_stride[:,ll]
                peaks, props = find_peaks(distance, prominence=2, distance=None)

                # Estimate baseline before each step using local minima
                inv_distance = -distance
                troughs, _ = find_peaks(inv_distance, prominence=2 / 2, distance=None)

                step_amplitudes = []
                step_times = []

                for peak in peaks:
                    # Find the closest following trough (baseline)
                    next_troughs = troughs[troughs > peak]
                    if len(next_troughs) == 0:
                        continue
                    baseline_idx = next_troughs[0]
                    amplitude = distance[peak] - distance[baseline_idx]
                    step_amplitudes.append(amplitude)
                    step_times.append(time[peak])

                step_amplitudes = np.array(step_amplitudes)
                step_times = np.array(step_times)

                self.stride_amp.append(step_amplitudes)
                self.stride_time.append(step_times)


                # Compute step frequency (Hz) in running 2 second window
                window = 2
                freqs = np.full(len(time), np.nan)  # preallocate

                for i, t in enumerate(time):
                    # count steps within [t - window/2, t + window/2]
                    mask = (step_times >= t - window/2) & (step_times <= t + window/2)
                    steps_in_window = step_times[mask]

                    if len(steps_in_window) >= 1:
                        intervals = np.diff(steps_in_window)
                        freqs[i] = 1 / np.mean(intervals)
                    else:
                        freqs[i] = np.nan

                self.stride_freq[:,ll] = freqs


            fig,ax = plt.subplots(5, 1, figsize=(16, 16))
            # rod speed
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 2, stride of hand
            ax[1].plot(self.t_interp, self.filtered_stride[:,0])
            ax[1].plot(self.t_interp , self.filtered_stride[:,1])
            ax[1].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Distance between left/right hand and the rod')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 3 foot
            ax[2].plot(self.t_interp, self.filtered_stride[:,2])
            ax[2].plot(self.t_interp, self.filtered_stride[:,3])
            ax[2].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].set_title('Distance between left/right foot and the rod')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 hand amplitude
            ax[3].stem(self.stride_time[0], self.stride_amp[0], linefmt='C0-',  basefmt=" ", label='left hand')
            ax[3].stem(self.stride_time[1], self.stride_amp[1],linefmt='C1-',  basefmt=" ",label='right hand')
            ax[3].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[3].set_title('Hand step amplitude')
            ax[3].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 (Third row, first column)
            ax[4].stem(self.stride_time[2], self.stride_amp[2], linefmt='C0-',  basefmt=" ", label='left foot')
            ax[4].stem(self.stride_time[3], self.stride_amp[3], linefmt='C1-',  basefmt=" ", label='right foot')
            ax[4].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[4].set_title('Foot step amplitude')

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
            
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis,'Stride amplitude.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            data = {'left hand': self.filtered_stride[:,0],
                    'right hand': self.filtered_stride[:,1],
                    'left foot': self.filtered_stride[:, 2],
                    'right foot': self.filtered_stride[:, 3],
                    'stride amplitude': self.stride_amp,
                    'stride time': self.stride_time,
                    'stride frequency': self.stride_freq,
                    'time': self.t_interp}
            #dataDF = pd.DataFrame(data)
            #dataDF.to_csv(savedatapath)
            # save to pickle file
            with open( os.path.join(self.analysis, 'stride_freq.pickle'), 'wb') as f:
                pickle.dump(data, f)

            #%%
            # cumulative area under the curve
            # cum_xcorr_foot = np.cumsum(xcorr['left foot-right foot'])/fps
            # cum_xcorr_hand = np.cumsum(xcorr['left hand-right hand']) / fps
            # cum_xcorr_left = np.cumsum(xcorr['left hand-left foot'])/fps
            # cum_xcorr_right = np.cumsum(xcorr['right hand-right foot']) / fps
            # plt.figure()
            # plt.plot(self.t_interp, cum_xcorr_foot)
            # plt.plot(self.t_interp, cum_xcorr_hand)
            # plt.plot(self.t_interp, cum_xcorr_left)
            # plt.plot(self.t_interp, cum_xcorr_right)
            # plt.plot(self.data['rodT'][timeMaskRod], self.data['rodSpeed_smoothed'][timeMaskRod])
            # plt.xlabel('time')
            # plt.ylabel('Cumulative area under the curve of xcorr')
            # plt.legend(['feet','hands','left','right', 'Rod speed'])
            # plt.savefig(os.path.join(self.analysis,'Stride correlation.png'), dpi=300)  # Save as PNG fil
            # #plt.show()
            # plt.close()
            # # cross correlation in 10 second window

            # # save data in csv
            # xcorr.to_csv(os.path.join(self.analysis, 'Stride crosscorrelation.csv'))

            #
            #%% tail angle
            # calculate spine 3 - tail 1 - tail 2 angle
            A = np.array([self.data['spine 3']['x'], self.data['spine 3']['y']]).T
            B = np.array([self.data['tail 1']['x'], self.data['tail 1']['y']]).T
            C = np.array([self.data['tail 2']['x'], self.data['tail 2']['y']]).T

            A = A[timeMaskDLC,:]
            B = B[timeMaskDLC,:]
            C = C[timeMaskDLC,:]
            # Calculate vectors AB and BC
            AB = B - A
            BC = C - B

            # Calculate the angle between AB and BC
            # Calculate dot and cross products for each time point
            dot_product = np.sum(AB * BC, axis=1)  # Dot product for each row (time point)
            cross_product = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]  # Cross product for each time point

            # Calculate the angle at each time point
            angles = np.arctan2(cross_product, dot_product)

            # Convert to degrees
            angles = np.degrees(angles)

            # interpolate and filter the angle
            fps = 50
            self.filtered_angle = np.full((len(self.t_interp)), np.nan)
            self.interp_angle = np.full((len(self.t_interp)), np.nan)

            # need to determine the cutoff frequency here

                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
            self.interp_angle= np.interp(self.t_interp, time_include, angles)

            self.filtered_angle = butter_lowpass_filter(self.interp_angle, 5,fps,order=5)


            # save data in csv
            tail_angle= pd.DataFrame({'angle':self.filtered_angle, 'time':self.t_interp})
            tail_angle.to_csv(os.path.join(self.analysis, 'Tail angle.csv'))
            # Calculate the angle in radians using atan2 for correct sign

            # plot the video frame with keypoint estimatino
            # frame_num = 7760
            # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
            # plt.figure()
            # plt.imshow(curr_frame)
            # kp_plot = ['tail 2']
            # for kp in kp_plot:
            #     plt.scatter(self.data[kp]['x'][frame_num], self.data[kp]['y'][frame_num], s=20)

            #%% head angle

            #%% tail position
            # plot the density distribution of the tail
            # set coordinate of tail 1 to be (0, 0)
            # ego_tail = {}
            # tail_key = ['tail 1', 'tail 2', 'tail 3']
            # for t in tail_key:
            #     ego_tail[t] = {}
            #     ego_tail[t]['x']= np.array(self.data[t]['x'])-np.array(self.data['tail 1']['x'])
            #     ego_tail[t]['y'] = np.array(self.data[t]['y']) - np.array(self.data['tail 1']['y'])
            #
            # plt.figure(figsize=(12, 6))
            # # Density plot for aligned b coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 2']), x='x', y='y',
            #             fill=True, cmap='Blues', alpha=0.5, label='Point B',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm())
            # # Density plot for aligned c coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 3']), x='x', y='y',
            #             fill=True, cmap='Reds', alpha=0.5, label='Point C',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm()
            #             )
            # plt.axhline(0, color='black', lw=1, ls='--', label='y = 0')
            # plt.axvline(0, color='black', lw=1, ls='--', label='x = 0')
            #
            # plt.title('Density Distribution of Aligned Points B and C')
            # plt.xlabel('Aligned B X')
            # plt.ylabel('Aligned B Y / Aligned C Y')
            # plt.axhline(0, color='black', lw=0.5, ls='--')
            # plt.axvline(0, color='black', lw=0.5, ls='--')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # with open(savedatapath, 'wb') as f:
            #     pickle.dump(self.stride, self. f)
            # f.close()
        else:
            print("Analysis already done")
            return np.nan

    def get_result(self):

        # go over the behavior result and get time before fell
        x=1
        rodData = read_rotarod_csv()

    def get_angular_velocity_running(self, t, savefigpath):
        # calculate angular velocity with running window t
        savedatapath = os.path.join(self.analysis, 'angular_velocity_running.pickle')

        if not os.path.exists(savedatapath):
            self.angVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.headAngVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.angVel_running[ff] = np.nanmean(self.angVel[ff:ff+t*self.fps])
                self.headAngVel_running[ff] = np.nanmean(self.headAngVel[ff:ff+t*self.fps])

            # plot the velocity here
            angPlot = StartPlots()
            angPlot.ax.plot(self.t[0:self.nFrames - 1 - t*self.fps], self.angVel_running)
            angPlot.ax.set_ylabel('Angular velocity')
            ax2 = angPlot.ax.twinx()
            ax2.plot(self.t[0:self.nFrames - 1 - t*self.fps], self.headAngVel_running, color='red')
            ax2.set_ylabel('Head angular velocity', color='red')
            angPlot.ax.set_xlabel('Time (s)')

            angPlot.save_plot('Running angular vel.tiff', 'tiff', savefigpath)
            plt.close(angPlot.fig)

            angVel_running = self.angVel_running
            headAngVel_running = self.headAngVel_running
            with open (savedatapath, 'wb') as f:
                pickle.dump([angVel_running,headAngVel_running], f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel_running, self.headAngVel_running = pickle.load(f)
            f.close()

    def get_angle(self, v1, v2):
        # get angle between two vectors
            v1_u = self.unit_vector(v1)
            v2_u = self.unit_vector(v2)

            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            if v1[0] * v2[1] - v1[1] * v2[0] < 0:
                angle = -angle
            return angle

    def plot_keypoints(self, nFrame):
        bodyparts = self.data['bodyparts']
        skeleton = [
            ['nose', 'head'],
            ['head', 'left ear'],
            ['head', 'right ear'],
            ['head', 'spine 1'],
            ['spine 1', 'left hand'],
            ['spine 1', 'right hand'],
            ['spine 1', 'spine 2'],
            ['spine 2', 'spine 3'],
            ['spine 3', 'left foot'],
            ['spine 3', 'right foot'],
            ['spine 3', 'tail 1'],
            ['tail 1', 'tail 2'],
            ['tail 2', 'tail 3']
        ]
        image = read_video(self.videoPath, nFrame, ifgray=True)
        plt.imshow(image)
        for bd in bodyparts:
            plt.scatter(self.data[bd]['x'][nFrame], self.data[bd]['y'][nFrame])
        for sk in skeleton:
            plt.plot([self.data[sk[0]]['x'][nFrame],self.data[sk[1]['x'][nFrame]]], [self.data[sk[0]]['y'][nFrame],self.data[sk[1]['y'][nFrame]]])

        plt.show()

    def unit_vector(self, v):
        """ Returns the unit vector of the vector.  """
        return v / np.linalg.norm(v)


if __name__ == "__main__":
    root_dir = r'Y:\HongliWang\Miniscope\ASD'

    #%% test code for odor behavior
    Odor = BehDataOdor(root_dir)

    # #%% load matlab code
    #eng = matlab.engine.start_matlab()

    # code_folder = r'C:\Users\Linda\Documents\GitHub\ASD_RLWM'
    #eng.addpath(eng.genpath(code_folder), nargout=0)

    # # read the data and save them to csv files
    # Odor.load_data()

    #Odor.session_behavior()

    #%% for behavior recordings
    # run it separately, if calcium imaging exist, then run the align_timeStamps in Imaging_pipeline


    #%% test code for rotarod behavior
    rotarod = BehDataRotarod(root_dir)

    rotarod.load_data()
