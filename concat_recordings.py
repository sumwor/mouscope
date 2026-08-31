# some sessions were recorded in multiple files
# due to the camera connection lost 
# reconcatenate the recordings into one file
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()

from utils_beh import extract_behavior_df
from utils_imaging import iso_to_timeofday, AI_timeStamp_correction

dataPath = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC003\ASDC003_260812'

AITimeStamps = [f for f in os.listdir(dataPath) if 'AITimeStamp' in f]
AIFiles = [f for f in os.listdir(dataPath) if 'AITTL' in f]
matFiles = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC003\ASDC003_20260812_AB.mat'

behDF = extract_behavior_df(matFiles)

AI_channels = 2
AI_freq = 1000

AI_matrix_1 = np.fromfile(os.path.join(dataPath, AIFiles[0]), dtype=np.float32)
AI_TimeStamp_1 = pd.read_csv(os.path.join(dataPath, AITimeStamps[0]), header=None).values.squeeze()  # unit in ms
# unit in ms

## lunghao code for AI timestamp correction
AI_TS_interp = AI_timeStamp_correction(AI_TimeStamp_1)

# rearange AI_matrix to two channels (one is ground)
AI_matrix_1 = AI_matrix_1.reshape(-1, AI_channels)
# look for rising edges of high voltage and get the time every 3 events
is_high = AI_matrix_1[:,0] > 4
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