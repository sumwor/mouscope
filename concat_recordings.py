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

#%% load data from separate chunks
dataPath = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC003\ASDC003_260812'

AITimeStamps = [f for f in os.listdir(dataPath) if 'AITimeStamp' in f]
AIFiles = [f for f in os.listdir(dataPath) if 'AITTL' in f]
matFiles = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC003\ASDC003_20260812_AB.mat'

videoTimeStamp = [f for f in os.listdir(dataPath) if 'ImgTimeStamp' in f]

behDF = extract_behavior_df(matFiles)

AI_channels = 2
AI_freq = 1000
n_valid_sum = 0
for AI_file in AIFiles:
    AI_matrix = np.fromfile(os.path.join(dataPath, AI_file))
    AI_matrix = AI_matrix.reshape(-1, AI_channels)
    is_high = AI_matrix[:,0] > 4
    edges = np.diff(is_high.astype(int))
    rising = np.where(edges == 1)[0] + 1
    falling = np.where(edges == -1)[0] + 1
    durations = (falling - rising) / AI_freq
    # exclude durations longer than 0.2 seconds (manual valve opening)
    valid_pulses = durations < 0.2
    n_valid_events = np.sum(valid_pulses)
    n_valid_sum += n_valid_events


# plt.figure()
# plt.plot(AI_matrix_2[:,1])
## lunghao code for AI timestamp correction



gaps = (rising_1[1:] - falling_1[:-1])/AI_freq

# For each pulse:
# prev_gap[i] = gap from previous pulse
# next_gap[i] = gap to next pulse

prev_gap = np.r_[np.inf, gaps]
next_gap = np.r_[gaps, np.inf]

# A pulse is "isolated" if it is >1 s away from both
# available neighbors.
isolated_idx = np.where(
    (prev_gap > 1) & (next_gap > 1)
)[0]

# if the isolated pulse is in the very beginning, set to 0
if len(isolated_idx) > 0 and isolated_idx[0] == 0:
    AI_matrix_1[rising_1[0]:falling_1[0],0] = 0
    # reshape the matrix back to 1-D
    AI_matrix_1D = AI_matrix_1.reshape(-1)


# read behavior csv files

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

#%% concatenate filesAI_matrix = np.concatenate([AI_matrix_1, AI_matrix_2], axis=0)

output_file = os.path.join(dataPath, 'ASDC004_260812_AITTL_concat')
AI_matrix = np.concatenate([AI_matrix_1D, AI_matrix_2], axis=0)
AI_matrix.tofile(output_file)

# concat AI_timestamp, probably alignment issue
AI_timestamp_concat = np.concatenate([AI_TimeStamp_1, AI_TimeStamp_2])
# save to csv
output_file_timestamp = os.path.join(dataPath, 'ASDC004_260812_AITimeStamp_concat.csv')
pd.DataFrame(AI_timestamp_concat).to_csv(output_file_timestamp, index=False, header=False)

# concat Imaging video timestamp
videoTimestamp_df = pd.DataFrame()
for ts in videoTimeStamp:
    df = pd.read_csv(os.path.join(dataPath, ts), header=None)
    videoTimestamp_df = pd.concat([videoTimestamp_df, df], ignore_index=True)
output_file_video_timestamp = os.path.join(dataPath, 'ASDC004_260812_ImgTimeStamp_concat.csv')
videoTimestamp_df.to_csv(output_file_video_timestamp, index=False, header=False)

# concat videos 
ImgVideoFiles = [f for f in os.listdir(dataPath) if 'ImgVideo' in f]
import subprocess

# Create concat list
list_file = os.path.join(dataPath, 'video_concat_list.txt')

with open(list_file, 'w') as f:
    for video in ImgVideoFiles:
        path = os.path.abspath(os.path.join(dataPath, video))
        f.write(f"file '{path}'\n")

output_file = os.path.join(dataPath, 'ASDC004_260812_ImgVideo_concat.avi')

subprocess.run([
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', list_file,
    '-c', 'copy',
    output_file
], check=True)

print(f'Saved: {output_file}')

# concat behavior video time stamp
behTimestamp_df = pd.DataFrame()
behTimestamp = [f for f in os.listdir(dataPath) if '.csv' in f and 'TimeStamp' not in f]
for ts in behTimestamp:
    df = pd.read_csv(os.path.join(dataPath, ts), header=None)
    behTimestamp_df = pd.concat([behTimestamp_df, df], ignore_index=True)
output_file_beh_timestamp = os.path.join(dataPath, 'ASDC004_260812_concat.csv')
behTimestamp_df.to_csv(output_file_beh_timestamp, index=False, header=False)

# concat behavior video
behVideoFiles = [f for f in os.listdir(dataPath) if '.mp4' in f]

# Create concat list
list_file = os.path.join(dataPath, 'video_concat_list.txt')

with open(list_file, 'w') as f:
    for video in behVideoFiles:
        path = os.path.abspath(os.path.join(dataPath, video))
        f.write(f"file '{path}'\n")

output_file = os.path.join(dataPath, 'ASDC004_260812_concat.mp4')

subprocess.run([
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', list_file,
    '-c', 'copy',
    output_file
], check=True)

print(f'Saved: {output_file}')


#%% check if the concatenated files can be processed by the analysis pipeline
AI_TimeStamp = pd.read_csv(output_file_timestamp, header=None).values.squeeze()  # unit in ms
AI_TS_interp = AI_timeStamp_correction(AI_TimeStamp)

LC_Mask = np.logical_and(behDF['schedule'] == 1, behDF['reward'] > 0)
trialNumber = np.arange(behDF.shape[0])
LC_trialNum = trialNumber[LC_Mask]

#load concate AI matrix
AI_matrix= np.fromfile(output_file)
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


# look for left correct trials
nLeftCorrect = np.sum(np.logical_and(behDF['schedule'] == 1, behDF['reward'] > 0))

# make a plot, go over behDF, if a left choice reward = 3, count 3 high voltage event
# if a left choice reward = 2, count 2 high voltage event
nPulses = np.sum(behDF['reward'][np.logical_or(behDF['schedule']==1, behDF['schedule']==3)])

# if not nPulses == n_valid_events:
#     print(f"Session file {self.data_index['Animal'][ii]}_{self.data_index['Date'][ii]}")
#     print("Mismatching between AI pulses and left correct trials, check!!!")


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


behTimeStamp = pd.read_csv()
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