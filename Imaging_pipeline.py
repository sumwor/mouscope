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

            # concatenate file if more than one (preprocess)
            # if len(behRecording) > 1:
            #     splited_folder = os.path.join(ImagingFolder, 'splited_session')
            #     os.makedirs(splited_folder, exist_ok=True)

            #     beh_concat_list = []
            #     for csv_file in behTimeStamp:
            #         df = pd.read_csv(csv_file, header=None)
            #         beh_concat_list.append(df)

            #         # move original file to splited_session
            #         shutil.move(csv_file, os.path.join(splited_folder, os.path.basename(csv_file)))
            #         beh_concat = pd.concat(beh_concat_list, ignore_index=True)
            #         concat_file = os.path.join(ImagingFolder, 'behTimeStamp_concatenated.csv')
            #         behTimeStamp = [concat_file]
            #         beh_concat.to_csv(concat_file, index=False)

            #     # imaging time stamp
            #     img_concat_list = []
            #     for csv_file in ImgTimeStamp:
            #         df = pd.read_csv(csv_file, header=None)
            #         img_concat_list.append(df)

            #         # move original file to splited_session
            #         shutil.move(csv_file, os.path.join(splited_folder, os.path.basename(csv_file)))
            #     img_concat = pd.concat(img_concat_list, ignore_index=True)
            #     concat_file = os.path.join(ImagingFolder, 'xxx_ImgTimeStamp_concatenated.csv')
            #     ImgTimeStamp = [concat_file]
            #     img_concat.to_csv(concat_file, index=False)

            #     # AI file
            #     AI_matrix_list = []
            #     AI_ts_list = []
            #     for mfile, tsfile in zip(AIMatrix, AITimsStamp):
            #         m = np.fromfile(mfile)
            #         t = pd.read_csv(tsfile, header=None).values.squeeze()
            #         AI_matrix_list.append(m)
            #         AI_ts_list.append(t)

            #         # move original files
            #         shutil.move(mfile, os.path.join(splited_folder, os.path.basename(mfile)))
            #         shutil.move(tsfile, os.path.join(splited_folder, os.path.basename(tsfile)))

            #     # concatenate
            #     AI_matrix_concat = np.concatenate(AI_matrix_list)
            #     AI_ts_concat = np.concatenate(AI_ts_list)

            #     # save concatenated AI files
            #     AI_matrix_file = os.path.join(ImagingFolder, 'AIMatrix_AITTL_concatenated.bin')
            #     AI_TS_file = os.path.join(ImagingFolder, 'xxx_AITimeStamp_concatenated.csv')
            #     AI_matrix_concat.tofile(AI_matrix_file)
            #     pd.DataFrame(AI_ts_concat).to_csv(AI_TS_file, index=False)
            #     AIMatrix = [AI_matrix_file]
            #     AITimsStamp = [AI_TS_file]


            # add these to the data_index
            self.data_index.loc[ii,'ROIFile'] = ROIFile[0] if ROIFile else None
            self.data_index.loc[ii,'behRecording'] = behRecording[0] if behRecording else None
            self.data_index.loc[ii,'behTimeStamp'] = behTimeStamp[0] if behTimeStamp else None
            self.data_index.loc[ii,'AIMatrix'] = AIMatrix[0] if AIMatrix else None
            self.data_index.loc[ii,'AITimsStamp'] = AITimsStamp[0] if AITimsStamp else None
            self.data_index.loc[ii,'ImgTimeStamp'] = ImgTimeStamp[0] if ImgTimeStamp else None
            self.data_index.loc[ii,'ifCalImg'] = ifCalImg
            self.data_index.loc[ii,'ifBehRecording'] = ifBehRecording


    def align_timeStamps(self):
        # files to align
        # read TTL pulse, TTL timestamp, image timestamp, behTimeStamp and align them with behavior csv file

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
                
                ## lunghao code for AI timestamp correctio
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



                plt.figure()
                plt.scatter(behDF['side_in'][LC_Mask], AI_TS_interp[matched]/1000)
                # plot the diagonal line
                plt.plot([behDF['side_in'][LC_trialNum[0]], behDF['side_in'][LC_trialNum[-1]]],
                        [AI_TS_interp[matched[0]]/1000, AI_TS_interp[matched[-1]]/1000],
                        'r--')
                # x label
                plt.xlabel('Left correct from behavior')
                plt.ylabel('Left correct from NI DAQ')
                plt.title('Time stamp alignment behavior vs. AI')
                savefigpath = os.path.join(self.analysis, self.data_index['Animal'][ii], self.behavior, 'Imaging',
                                            self.data_index['Date'][ii])
                if not os.path.exists(savefigpath):
                    os.makedirs(savefigpath)
                plt.savefig(os.path.join(savefigpath, 'TimeStamp_alignment.png'))
                plt.close()
                            
                t_offset = AI_TS_interp[matched]/1000 - behDF['side_in'][LC_trialNum]
            
            # load behavior recording timestamp if exists
            if os.path.exists(self.data_index['behTimeStamp'][ii]):
                behTimeStamp = pd.read_csv(self.data_index['behTimeStamp'][ii], header=None)
                header = ['TimeStamp']
                behTimeStamp.columns = header
                behTimeStamp['AlignedTimeStamp'] = behTimeStamp['TimeStamp']/1000 - t_offset
                old_path = self.data_index['behTimeStamp'][ii]
                folder, old_file = os.path.split(old_path)
                new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                behTimeStamp.to_csv(new_file, index=False)
            


            if os.path.exists(self.data_index['ImgTimeStamp'][ii]):
                ImgTimeStamp = pd.read_csv(self.data_index['ImgTimeStamp'][ii][cc], header=None)
                # define headers
                header = ['TimeStamp', 'FrameNumber', 'TTL', 'W', 'X', 'Y', 'Z']
                ImgTimeStamp.columns = header
                            
                # convert absolute time stamp (first column) to total minisecond, timeofday
                ImgTimeStamp['AlignedTimeStamp']= ImgTimeStamp['TimeStamp'].apply(iso_to_timeofday)-t_offset
                old_path = self.data_index['ImgTimeStamp'][ii]
                folder, old_file = os.path.split(old_path)
                new_file = os.path.join(folder, old_file[:-4] + "_aligned.csv")
                ImgTimeStamp.to_csv(new_file, index=False)

        

if __name__ == "__main__":

    # use interactive matplotlib backend

    plt.ion()

    root_dir = r'Y:\HongliWang\Miniscope\ASD'

    Odor_Beh = BehDataOdor(root_dir)

    imaging_data = Imaging(root_dir, Odor_Beh)