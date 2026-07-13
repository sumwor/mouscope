# code to process behavior files and align calcium data with behavior timestamps
import copy
import glob
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import imageio.v3 as iio
import cv2

# matplotlib
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.stats.api as smf

from scipy.optimize import minimize
from scipy.signal import correlate, find_peaks, hilbert, spectrogram
from scipy.special import expit
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon
from statsmodels.stats.multitest import multipletests

# Project-local utilities
from pyPlotHW import StartPlots
from utils_beh import *
from utils_imaging import *
from utils_model import *
from utils_Deeplabcut import *
# add matlab code into the path
#eng.addpath(r'C:\Users\Linda\Documents\GitHub\ASD_RLWM\Behavior', nargout=0)

class BehData:

    def __init__(self, root_path, strain):
        self.root_path = root_path
        self.strain = strain
        self.data = os.path.join(self.root_path, 'Data')
        self.analysis = os.path.join(self.root_path, 'Analysis')
        self.summary = os.path.join(self.root_path, 'Summary')
        self.AnimalInfo = pd.read_csv(os.path.join(self.data, 'AnimalList.csv'))
        self.Animals = [str(x) for x in self.AnimalInfo['AnimalID']]
        self.Genotypes = self.AnimalInfo['Genotype']
        if 'Gender' in self.AnimalInfo.columns:
            self.Gender = self.AnimalInfo['Gender']
        else:
            self.Gender = ['M']*len(self.Animals)
        if 'Cells' in self.AnimalInfo.columns:
            self.ImageCell = self.AnimalInfo['Cells']
        else:
            self.ImageCell = [None] * len(self.Animals)
        if 'hemisphere' in self.AnimalInfo.columns:
            self.Hemisphere = self.AnimalInfo['hemisphere']
        else:   
            self.Hemisphere = [None] * len(self.Animals)

        strain_parts = strain.split('_')
        if any(part in ['Cntnap2'] for part in strain_parts):
            self.Mut = 'KO'
        elif any(part in ['TSC2', 'Shank3B', 'ChD8', 'Syngap', 'Scn2A'] for part in strain_parts):
            self.Mut = 'HET'
        elif any(part in ['Nlgn3'] for part in strain_parts):
            self.Mut = 'HEM'

class BehDataOF(BehData):

    def __init__(self, root_file, strain):
        super().__init__(root_file, strain)
        self.bodyparts = ['nose', 'head', 'left ear', 'right ear', 'left hand', 'right hand',
                          'spine 1', 'spine 2', 'spine 3', 'left foot', 'right foot', 'tail 1',
                          'tail 2', 'tail 3']
        # load animalID
        
        self.make_dataIndex()
        self.behavior = 'Openfield'

        # get the behCSV path
        #self.load_data()

    def make_dataIndex(self):
        # create a dataIndex for all open field data
        DLC_results = []
        video = []
        timestamp = []
        animalID = []
        analysis = []
        GeneBGID = []
        sessionID = []
        genderID = []
        behAge = []
        for aidx,aa in enumerate(self.Animals):
            sessionPattern = r'_([0-9]{1,2})(?=DLC)'
            filePatternCSV = '*' + aa + '*_OF*filtered.csv'
            filePatternVideo = '*' + aa + '*.avi'
            filePatternTimeStamp = '*' + aa + '*timestamps*.csv'
            dataFolder = os.path.join(self.data, aa, 'Openfield', 'BehavioralRecording')
            # find folders in dataFolder
            for folder in os.listdir(dataFolder):
                if not folder == '.DS_Store':
                    csvfiles = glob.glob(f"{dataFolder}/{folder}/{filePatternCSV}")
                    DLC_results.append(csvfiles[0])
                    video.append(glob.glob(f"{dataFolder}/{folder}/{filePatternVideo}")[0])
                    timestamp.append(glob.glob(f"{dataFolder}/{folder}/{filePatternTimeStamp}")[0])
                    animalID.append(aa)
                    analysis.append(os.path.join(self.analysis, aa, 'Openfield', folder))
                    GeneBGID.append(self.Genotypes[aidx])
                    genderID.append(self.Gender[aidx])

                    # find the recorded age
                    behAge.append(folder[-3:])

        self.data_index = pd.DataFrame(animalID, columns=['Animal'])
        self.data_index['CSV'] = DLC_results
        self.data_index['Video'] = video
        self.data_index['Age'] = behAge
        self.data_index['Timestamp'] = timestamp
        self.data_index['AnalysisPath'] = analysis
        self.data_index['Genotype'] = GeneBGID
        self.data_index['Gender'] = genderID
        self.nSubjects = len(self.Animals)

        self.nSessions = len(self.data_index['Animal'])
        DLC_obj = []

        minFrames = 10 ** 8
        for s in range(self.nSessions):
            analysisPath = self.data_index['AnalysisPath'][s]
            filePath = self.data_index['CSV'][s]
            videoPath = self.data_index['Video'][s]
            fps = 30
            dlc = DLCSession(filePath, videoPath, None, analysisPath, fps)
            DLC_obj.append(dlc)
            if dlc.nFrames < minFrames:
                minFrames = dlc.nFrames

        self.minFrames = minFrames
        self.data_index['DLC_obj'] = DLC_obj
        self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSessions)

        self.WTIdx = animalIdx[self.data_index['Genotype'] == self.WTGene]
        self.MutIdx = animalIdx[self.data_index['Genotype'] == self.mutGene]
        # grouping the animals

        if len(np.unique(self.Gender))==2:
            self.maleIdx = np.where(self.data_index['Gender']=='M')[0]
            self.femaleIdx = np.where(self.data_index['Gender']=='F')[0]

    def center_analysis(self, savefigpath):
        centerMat = np.full((self.minFrames, self.nSubjects), np.nan)
        runningAve_center = np.full((self.minFrames, self.nSubjects), np.nan)
        numCrossMat = np.full((self.minFrames, self.nSubjects), np.nan)
        plotT = np.arange(self.minFrames)/self.fps
        for idx, obj in enumerate(self.data['DLC_obj']):
            savefigFolder = os.path.join(self.analysisFolder, self.Animals[idx])
            if not os.path.exists(savefigFolder):
                os.makedirs(savefigFolder)
            obj.moving_trace(savefigFolder)
            obj.get_time_in_center()
            t = 5*60
            obj.plot_distance_to_center(t, savefigFolder)
            centerMat[:,idx] = obj.cumu_time_center[0:self.minFrames]
            numCrossMat[:, idx] = obj.num_cross[0:self.minFrames]
            if idx==0:
                nbins = len(obj.dist_center_bins[1])
                centerDistMat = np.full((nbins-1, self.nSubjects), np.nan)
                centerDistMat30 = np.full((nbins - 1, self.nSubjects), np.nan)
            centerDistMat[:,idx] = obj.dist_center_bins[0]
            centerDistMat30[:,idx] = obj.dist_center_bins_30[0]

            runningAve_center[0: len(obj.dist_center_running), idx]=obj.dist_center_running.flatten()

        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)
        # save centerMat result

        # total time in the center
        totalCenter = centerMat[-1,:]
        totalCross = numCrossMat[-1,:]
        # violin plot
        custom_palette = {0: WTColor, 1: MutColor}
        ax=sns.violinplot(data=[totalCenter[self.WTIdx], totalCenter[self.MutIdx]],palette=custom_palette)
        ax.set_xticklabels(['WT', 'Mut'])
        ax.set_ylabel('Total time in the center')
        ax.set_xlabel('Group')
        ax.set_title('Total time in the center')
        plt.savefig(savefigpath + '\\violin_center_time.png', dpi=300)
        plt.savefig(savefigpath + '\\violin_center_time.svg', dpi=300)
        plt.close()

        ax=sns.violinplot(data=[totalCross[self.WTIdx], totalCross[self.MutIdx]],palette=custom_palette)
        ax.set_xticklabels(['WT', 'Mut'])
        ax.set_ylabel('Total cross time')
        ax.set_xlabel('Group')
        ax.set_title('Total cross time')
        plt.savefig(savefigpath + '/violin_cross_time.png', dpi=300)
        plt.savefig(savefigpath + '/violin_cross_time.svg', dpi=300)
        plt.close()

        data = {'animalID':self.Animals,
                'timeinCenter': totalCenter,
                'crossTime': totalCross}
        data = pd.DataFrame(data)
        data.to_csv(savefigpath + '/timeinCenter.csv')

        # WTBoot = bootstrap(centerMat[:, self.WTIdx], 1,
        #                        centerMat[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(centerMat[:, self.MutIdx], 1,
        #                         centerMat[:, self.MutIdx].shape[0])
        # WTColor = (255 / 255, 189 / 255, 53 / 255)
        # MutColor = (63 / 255, 167 / 255, 150 / 255)
        #
        # binX = (obj.dist_center_bins[1][0:-1] + obj.dist_center_bins[1][1:]) / 2
        #
        # for ss in ['male','female','allsex']:
        #     # plot distance
        #     self.plot_movement_results(centerMat,plotT,savefigpath,
        #                                'Time spent in the center', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(centerDistMat,binX,savefigpath,
        #                                'Distribution of distance from center', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(centerDistMat30,binX,savefigpath,
        #                                'Distribution of distance from center 30 mins', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(runningAve_center,plotT,savefigpath,
        #                                'Time spent in the center in running 5 mins windows', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(numCrossMat,plotT,savefigpath,
        #                                'Num of crossings', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)

        # KS test
        # from scipy.stats import ks_2samp
        # WTMale = centerDistMat30[:,  list(set(self.WTIdx) & set(self.maleIdx))]
        # MutMale = centerDistMat30[:,  list(set(self.MutIdx) & set(self.maleIdx))]
        # WTFemale = centerDistMat30[:,  list(set(self.WTIdx) & set(self.femaleIdx))]
        # MutFemale = centerDistMat30[:,  list(set(self.MutIdx) & set(self.femaleIdx))]
        # # Assuming you have two arrays of data: data1 and data2
        # # Perform the KS test
        # statistic, p_value = ks_2samp(WTMaleBoot['bootAve'], MutMaleBoot['bootAve'])
        #
        # from scipy.stats import permutation_test
        # res = permutation_test((WTMale.T,MutMale.T), ks_2samp)
        #
        # # Print the test statistic and p-value
        # print("KS statistic:", statistic)
        # print("p-value:", p_value)
        # # two-way ANOVA for centerDistMat30
        # gene_anova_male = []
        # dist_anova_male = []
        # response_anova_male = []
        # subject_male = []
        # gene_anova_female = []
        # dist_anova_female = []
        # subject_female = []
        # response_anova_female = []
        #
        # for t in range(len(self.GeneBG)):
        #     for s in range(len(binX)):
        #         if self.Sex[t] == 'M':
        #             response_anova_male.append(centerDistMat30[s,t])
        #             gene_anova_male.append(self.GeneBG[t])
        #             dist_anova_male.append(binX[s])
        #             subject_male.append(self.animals[t])
        #         else:
        #             response_anova_female.append(centerDistMat30[s,t])
        #             gene_anova_female.append(self.GeneBG[t])
        #             dist_anova_female.append(binX[s])
        #             subject_female.append(self.animals[t])
        #
        # anova_data = pd.DataFrame({'gene': gene_anova_male,
        #                            'dist': dist_anova_male,
        #                            'response': response_anova_male,
        #                            'subject': subject_male
        #                            })
        # model = ols('response ~ gene + dist + gene:dist', anova_data).fit()
        # anova_table = sm.stats.anova_lm(model, typ=3)
        # # print ANOVA table
        # print(anova_table)
        #
        # # three way anova?
        # gene_anova = []
        # dist_anova = []
        # response_anova = []
        # subject = []
        # sex = []
        #
        #
        # for t in range(len(self.GeneBG)):
        #     for s in range(len(binX)):
        #         response_anova.append(centerDistMat30[s,t])
        #         gene_anova.append(self.GeneBG[t])
        #         dist_anova.append(binX[s])
        #         subject.append(self.animals[t])
        #         sex.append(self.Sex[t])
        #
        # anova_data = pd.DataFrame({'gene': gene_anova,
        #                            'dist': dist_anova,
        #                            'response': response_anova,
        #                            'sex': sex,
        #                            'subject': subject
        #                            })
        # model = ols('response ~ gene + dist + sex + gene:dist + gene:sex + dist:sex + dist:sex:gene', anova_data).fit()
        # anova_table = sm.stats.anova_lm(model, typ=3)
        # # print ANOVA table
        # print(anova_table)
        #
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Time spent in the center (s)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Time spent in the center.tif', 'tif', savefigpath)
        # distPlot.save_plot('Time spent in the center.svg', 'svg', savefigpath)
        #
        # # distribution of distance from center
        # WTBoot = bootstrap(centerDistMat[:, self.WTIdx], 1,
        #                        centerDistMat[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(centerDistMat[:, self.MutIdx], 1,
        #                         centerDistMat[:, self.MutIdx].shape[0])
        # binX = (obj.dist_center_bins[1][0:-1] + obj.dist_center_bins[1][1:])/2
        # distPlot = StartPlots()
        # distPlot.ax.plot(binX, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(binX, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(binX, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(binX, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Distance from center (px)')
        # distPlot.ax.set_ylabel('Number of frames')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Distribution of distance from center.tif', 'tif', savefigpath)
        # distPlot.save_plot('Distribution of distance from center.svg', 'svg', savefigpath)
        #
        # # plot average distance from center in running windows
        # WTBoot = bootstrap(runningAve_center[:, self.WTIdx], 1,
        #                        runningAve_center[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(runningAve_center[:, self.MutIdx], 1,
        #                         runningAve_center[:, self.MutIdx].shape[0])
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Time spent in the center in running 5 mins windows (s)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Time spent in the center in running 5 mins windows.tif', 'tif', savefigpath)
        # distPlot.save_plot('Time spent in the center in running 5 mins windows.svg', 'svg', savefigpath)

    def motion_analysis(self, savefigpath=None):

        # basic analysis for motion related variables
        # distance traveled, speed, angular velocity...
        distanceMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        velocityMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        # in 5 mins window
        runningAve_distance = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        runningAve_velocity = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        #
        velEdges = np.arange(0, 1000, 10)
        velocityDist = np.full((len(velEdges), self.nSubjects), np.nan)
        angEdges = np.arange(-15, 15, 0.5)
        angularDist = np.full((len(angEdges), self.nSubjects), np.nan)
        headAngularDist = np.full((len(angEdges), self.nSubjects), np.nan)

        for idx, obj in enumerate(self.data_index['DLC_obj']):
            obj.get_movement()
            # cumulative curve of distance travelled
            cumu_dist = np.cumsum(obj.dist)
            distanceMat[:, idx] = cumu_dist[0:self.minFrames - 1]
            velocityMat[:, idx] = obj.vel[0:self.minFrames - 1, 0]
            counts, _ = np.histogram(obj.vel, bins=velEdges)
            velocityDist[0:-1, idx] = counts * 100 / (sum(counts))

            obj.get_angular_velocity()
            counts, _ = np.histogram(obj.angVel, bins=angEdges)
            angularDist[0:-1, idx] = counts * 100 / (sum(counts))

            obj.get_head_angular_velocity()
            counts, _ = np.histogram(obj.headAngVel, bins=angEdges)
            headAngularDist[0:-1, idx] = counts * 100 / (sum(counts))

            # running windows
            if savefigpath is None:
                savefigFolder = self.data_index.iloc[idx]['AnalysisPath']
            else:
                savefigFolder = os.path.join(savefigpath, str(self.Animals[idx]))
            os.makedirs(savefigFolder, exist_ok=True)
            t = 5*60  # running windos of 5 mins
            obj.get_movement_running(t, savefigFolder)
            obj.get_angular_velocity_running(t, savefigFolder)

            runningAve_distance[0:len(obj.dist_running),idx]=obj.dist_running.flatten()
            runningAve_velocity[0:len(obj.dist_running), idx] = obj.vel_running.flatten()
        """ make plots"""
        """distance plot"""
       
        mutLabel = self.mutGene

        # WTIdx = np.where(self.data['GeneBG'] == 'WT')[0]

        # plot the result without considering sex info
        # WTBoot = bootstrap(distanceMat[:, self.WTIdx], 1,
        #                        distanceMat[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(distanceMat[:, self.MutIdx], 1,
        #                         distanceMat[:, self.MutIdx].shape[0],500)
        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)

        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Distance travelled (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Distance traveled.tif', 'tif', savefigpath)
        # distPlot.save_plot('Distance traveled.svg', 'svg', savefigpath)
        #
        # """velocity plot"""
        # WTBoot = bootstrap(velocityDist[:, self.WTIdx], 1,
        #                        velocityDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(velocityDist[:, self.MutIdx], 1,
        #                         velocityDist[:, self.MutIdx].shape[0])
        # velPlot = StartPlots()
        # velPlot.ax.plot(velEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # velPlot.ax.fill_between(velEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # velPlot.ax.plot(velEdges, MutBoot['bootAve'], color=MutColor, label='KO')
        # velPlot.ax.fill_between(velEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # velPlot.ax.set_xlabel('Velocity (px/s)')
        # velPlot.ax.set_ylabel('Velocity distribution (%)')
        # velPlot.legend(['WT', 'KO'])
        # velPlot.save_plot('Velocity distribution.tif', 'tif', savefigpath)
        # velPlot.save_plot('Velocity distribution.svg', 'svg', savefigpath)
        #
        # """ plot angular velocity distribution"""
        # WTBoot = bootstrap(angularDist[:, self.WTIdx], 1,
        #                        angularDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(angularDist[:, self.MutIdx], 1,
        #                         angularDist[:, self.MutIdx].shape[0])
        #
        # """angular velocity plot"""
        # angPlot = StartPlots()
        # angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        # angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.set_xlabel('Angular velocity (radian/s)')
        # angPlot.ax.set_ylabel('Angular velocity distribution (%)')
        # angPlot.legend(['WT', 'Mut'])
        # angPlot.save_plot('Angular velocity distribution.tif', 'tif', savefigpath)
        # angPlot.save_plot('Angular velocity distribution.svg', 'svg', savefigpath)
        #
        # """plot head angular velocity distribution"""
        # WTBoot = bootstrap(headAngularDist[:, self.WTIdx], 1,
        #                        headAngularDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(headAngularDist[:, self.MutIdx], 1,
        #                         headAngularDist[:, self.MutIdx].shape[0])
        #
        # angPlot = StartPlots()
        # angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        # angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.set_xlabel('Angular velocity(head) (radian/s)')
        # angPlot.ax.set_ylabel('Angular velocity(head) distribution (%)')
        # angPlot.legend(['WT', 'Mut'])
        # angPlot.save_plot('Angular velocity(head) distribution.tif', 'tif', savefigpath)
        # angPlot.save_plot('Angular velocity(head distribution.svg', 'svg', savefigpath)
        #
        #
        # # distance and velocity in 5 mins running window
        # WTBoot = bootstrap(runningAve_distance[:, self.WTIdx], 1,
        #                        runningAve_distance[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(runningAve_distance[:, self.MutIdx], 1,
        #                         runningAve_distance[:, self.MutIdx].shape[0],500)
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Running average distance travelled in 5 mins (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Running average distance travelled in 5 mins.tif', 'tif', savefigpath)
        # distPlot.save_plot('Running average distance travelled in 5 mins.svg', 'svg', savefigpath)
        #
        # WTBoot = bootstrap(runningAve_velocity[:, self.WTIdx], 1,
        #                        runningAve_velocity[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(runningAve_velocity[:, self.MutIdx], 1,
        #                         runningAve_velocity[:, self.MutIdx].shape[0],500)
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Running average velocity in 5 mins (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Running average distance travelled in 5 mins.tif', 'tif', savefigpath)
        # distPlot.save_plot('Running average distance travelled in 5 mins.svg', 'svg', savefigpath)
        #
        # plt.close('all')

        # plot result separating male and female
        # save distanceMat, runningAve_distance

        # convert to cm
        if savefigpath is None:
            savefigpath = self.data_index.iloc[0]['AnalysisPath']
        os.makedirs(savefigpath, exist_ok=True)
        savedistPath = os.path.join(savefigpath, 'CumulativeDistance.csv')
        data = {}
        for idx,animal in enumerate(self.Animals):
            data[animal] = distanceMat[:,idx]
        data['time'] = self.plotT
        data = pd.DataFrame(data)
        data.to_csv(savedistPath)

        savedistPath = os.path.join(savefigpath, 'runningAverageDistance.csv')
        data = {}
        for idx,animal in enumerate(self.Animals):
            data[animal] = runningAve_distance[:,idx]
        data['time'] = self.plotT
        data = pd.DataFrame(data)
        data.to_csv(savedistPath)

        for ss in ['male','female', 'allsex']:
            # plot distance
            self.plot_movement_results(distanceMat,self.plotT,savefigpath,
                                       'Distance travelled', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)

            self.plot_movement_results(velocityDist,velEdges,savefigpath,
                                       'Velocity', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)
            self.plot_movement_results(angularDist,angEdges,savefigpath,
                                       'Angular velocity', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)
            self.plot_movement_results(runningAve_distance,self.plotT,savefigpath,
                                       'Distance running 5 mins', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)

    def plot_movement_results(self, variableMat, plotT, savefigpath, label, group, leg,color1, color2):
        if group =='male':
        # if consider sex info
            WTIdx = list(set(self.WTIdx) & set(self.maleIdx))
            mutIdx = list(set(self.MutIdx) & set(self.maleIdx))
        elif group == 'female':
            WTIdx = list(set(self.WTIdx) & set(self.femaleIdx))
            mutIdx = list(set(self.MutIdx) & set(self.femaleIdx))
        elif group == 'allsex':
            WTIdx = self.WTIdx
            mutIdx = self.MutIdx
        WTBoot = bootstrap(variableMat[:, WTIdx], 1,
                               variableMat[:, WTIdx].shape[0], 200)
        MutBoot = bootstrap(variableMat[:, mutIdx], 1,
                                variableMat[:, mutIdx].shape[0],200)
        WTColor = color1
        MutColor = color2

        distPlot = StartPlots()
        distPlot.ax.plot(plotT, WTBoot['bootAve'], color=color1, label='WT')
        distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
                                     WTBoot['bootHigh'], color=color1, alpha=0.2, label='_nolegend_')
        distPlot.ax.plot(plotT, MutBoot['bootAve'], color=color2, label='KO')
        distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
                                     MutBoot['bootHigh'], color=color2, alpha=0.2, label='_nolegend_')
        distPlot.ax.set_xlabel('Time (s)')
        title = label + ' ' + group
        distPlot.ax.set_ylabel(title)
        distPlot.legend(leg)
        #distPlot.ax.set_ylim(0, np.nanmax(variableMat))
        # save the plot
        distPlot.save_plot(title+'.png', 'png', savefigpath)
        distPlot.save_plot(title+'.svg', 'svg', savefigpath)
        plt.close()

class BehDataOdor(BehData):

    def __init__(self, root_file, strain):
        super().__init__(root_file, strain)
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
                    'Gender': self.Gender[aIdx],
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
                    resultdf = extract_behavior_df(beh)

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
            
            #eng.ASD_session(resultdf_path,self.data_index['Protocol'][ss],self.data_index['Animal'][ss], 
            #                self.data_index['Date'][ss],self.data_index['AnalysisPath'][ss],nargout=0)
            
    def session_analysis(self):
        # session-wise anlaysis
        # plot single session performance
        # model fitting
        # use the policy-gradient model for learnind window estimation (take the derivative)
        
        # go over each session
        nSessions = self.data_index.shape[0]
        for ss in range(nSessions):
            # extract the behavior data
            resultdf_path = self.data_index['BehCSV'][ss]
            resultdf = pd.read_csv(resultdf_path)
            
            #%%
            # 1. session performance
            protocol = self.data_index['Protocol'][ss]
            save_path = self.data_index['AnalysisPath'][ss]
            animalID = self.data_index['Animal'][ss]
            date = self.data_index['Date'][ss]
            protocolDay = self.data_index['ProtocolDay'][ss]
            label = f'{animalID}_{date}_{protocol}_{protocolDay}'
            plot_session(resultdf, protocol, save_path = save_path, label = label)

            # model fitting!
    
    def model_fitting(self, fit_mode):
        # fit computational models to the behavioral data
        # fit mode: 'session' or 'concat'
        #          'session': fit model to each session separately
        #          'concat': fit model to the concatenated data of all sessions
        

        if fit_mode == 'session':
            nSessions = self.data_index.shape[0]
            protocols = ['AB1', 'AB2', 'AB3', 'CD1', 'CD2', 'CD3']
            fit_params = pd.DataFrame()
            # store the value for psychometric curve
            fit_psychometric = {}
                        
            for pp in protocols:
                fit_psychometric[pp] = pd.DataFrame()
                fit_psychometric[pp]['genotype'] = np.array([])
                fit_psychometric[pp]['weighted_sum'] = np.array([])
                fit_psychometric[pp]['choice'] = np.array([])
                fit_psychometric[pp]['gender'] = np.array([])

            for ss in range(nSessions):
                # extract the behavior data
                fit_params.loc[ss, 'animal'] = self.data_index['Animal'][ss]
                fit_params.loc[ss, 'mode'] = fit_mode
                fit_params.loc[ss, 'genotype'] = self.data_index['Genotype'][ss]
                fit_params.loc[ss, 'gender'] = self.data_index['Gender'][ss]

                resultdf_path = self.data_index['BehCSV'][ss]
                resultdf = pd.read_csv(resultdf_path)
            
            #%% policy gradient model
                protocol = self.data_index['Protocol'][ss]
                protocolDay = self.data_index['ProtocolDay'][ss]
                animalID = self.data_index['Animal'][ss]
                save_path = os.path.join(self.data_index['AnalysisPath'][ss], 'latent')
                os.makedirs(save_path, exist_ok=True)
                fit_params.loc[ss, 'protocol'] = f'{protocol}{protocolDay}'

                savedatapath = os.path.join(save_path,'policy_gradient_fit.json')

                # preprocess the data (remove AB trials for AB-CD sessions)
                # fit AB and AB-CD sessions only
                if protocol == 'AB-CD-DC' or protocol=='AB-DC' or protocolDay > 3:
                    continue
                resultdf.replace({"actions": ["NAN","NaN", "nan", "None", ""]}, np.nan, inplace=True)
                resultdf.replace({"schedule": ["NAN", "NaN", "nan", "None", ""]}, np.nan, inplace=True)
                data = resultdf.dropna(subset=["actions"])
                data = data.dropna(subset=['schedule'])
                data.schedule = data.schedule.astype(int)
                # if CD session
                nOdors = np.unique(data['schedule'])
                if 'CD' in protocol:
                    # remove AB trials
                    data = data[np.logical_and(data['schedule']!=1,data['schedule']!=2)].copy()
                    data['schedule'] = data['schedule']-2


                # if os.path.exists(savedatapath):
                #     # load the existing fit
                #     with open(savedatapath, 'r') as f:
                #         latent_fit = json.load(f)
                #else:
                latent_fit = fit_policy_gradient(data,animalID=animalID, savedatapath=savedatapath)

                weights = latent_fit['weight']
                opt_vars = latent_fit['args']['optList']
                for widx, ww in enumerate(weights):
                    for vv in opt_vars:
                        fit_params.loc[ss, f'{ww}_{vv}'] = latent_fit['opt_hyper'][vv][widx]
                fit_params.loc[ss, f'AIC'] = latent_fit['AIC']
                fit_params.loc[ss, f'BIC'] = latent_fit['BIC']

                # for psychometric plot
                temp = pd.DataFrame()
                temp['weighted_sum'] = latent_fit['weighted_sum']
                temp['genotype'] = fit_params['genotype'][ss]
                temp['animal'] = fit_params['animal'][ss]
                temp['choice'] = latent_fit['args']['dat']['inputs']['cBoth']
                temp['gender'] = fit_params['gender'][ss]
                fit_psychometric[pp]= pd.concat(
                    (fit_psychometric[pp], temp))
                    
                model_label = 'Policy Gradient'
                savefigpath = os.path.join(save_path, f'{animalID}_{protocol}_latent_fit')
                plot_latent_session(data, latent_fit, model_label,savefigpath)

                

        elif fit_mode == 'concat':
            protocols = ['AB', 'CD']
            fit_params = pd.DataFrame()
            # store the value for psychometric curve
            fit_psychometric = {}
            
            for pp in protocols:
                fit_psychometric[pp] = pd.DataFrame()
                fit_psychometric[pp]['genotype'] = np.array([])
                fit_psychometric[pp]['weighted_sum'] = np.array([])
                fit_psychometric[pp]['choice'] = np.array([])
                fit_psychometric[pp]['gender'] = np.array([])

            for aidx, animal in enumerate(self.data_index['Animal'].unique()):
                fit_params.loc[aidx, 'animal'] = animal
                fit_params.loc[aidx, 'mode'] = 'concat'
                fit_params.loc[aidx, 'genotype'] = self.data_index[self.data_index['Animal'] == animal]['Genotype'].iloc[0]
                fit_params.loc[aidx, 'gender'] = self.data_index[self.data_index['Animal'] == animal]['Gender'].iloc[0]
            
                result_concat = {}
                result_concat['AB'] = pd.DataFrame()
                result_concat['CD'] = pd.DataFrame()

                Animal_sessions = self.data_index[self.data_index['Animal'] == animal]
                AB_CD_1_idx = Animal_sessions[Animal_sessions['Protocol'].str.contains('AB-CD')].index[0] if not Animal_sessions[Animal_sessions['Protocol'].str.contains('AB-CD')].empty else None            
                if AB_CD_1_idx is not None:
                    AB_sessions = Animal_sessions.loc[:AB_CD_1_idx-1]
                else:
                    AB_sessions = Animal_sessions
                AB_CD_sessions = Animal_sessions[np.logical_and(Animal_sessions['Protocol']=='AB-CD', 
                                                                Animal_sessions['ProtocolDay'] <=3 )]

                for sIdx in AB_sessions.index:
                    temp_result = pd.read_csv(AB_sessions.loc[sIdx]['BehCSV'])
                    # remove miss trials
                    temp_result = temp_result[~np.isnan(temp_result['actions'])]
                    result_concat['AB'] = pd.concat([result_concat['AB'], temp_result], ignore_index=True)

                for sIdx in AB_CD_sessions.index:
                    temp_result = pd.read_csv(AB_CD_sessions.loc[sIdx]['BehCSV'])
                    # remove miss trials
                    temp_result = temp_result[~np.isnan(temp_result['actions'])]
                    # remove AB trials
                    temp_result = temp_result[temp_result['schedule']>2]
                    result_concat['CD'] = pd.concat([result_concat['CD'], temp_result], ignore_index=True)
                    

                # calculate running reward rate
                for pp in protocols:
                    if len(result_concat[pp]) == 0:
                        continue
                    resultdf = result_concat[pp]
                    resultdf.replace({"actions": ["NAN","NaN", "nan", "None", ""]}, np.nan, inplace=True)
                    resultdf.replace({"schedule": ["NAN", "NaN", "nan", "None", ""]}, np.nan, inplace=True)
                    data = resultdf.dropna(subset=["actions"])
                    #data = data.dropna(subset=['schedule'])
                    data.schedule = data.schedule.astype(int)
                    if pp == 'CD':
                        data['schedule'] = data['schedule']-2
                    save_path = os.path.join(self.analysis,animal,self.behavior, 'Behavior', 'Summary')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    savedatapath = os.path.join(save_path,
                                                 f'{animal}_{pp}_fit.json')
                    if os.path.exists(savedatapath):
                    # load the existing fit
                        with open(savedatapath, 'r') as f:
                            latent_fit = json.load(f)
                    else:
                        latent_fit = fit_policy_gradient(data, 
                                                         animalID=animal, savedatapath=savedatapath)
                    
                    # load data in dataframe 
                    weights = latent_fit['weight']
                    opt_vars = latent_fit['args']['optList']
                    for widx, ww in enumerate(weights):
                        for vv in opt_vars:
                            fit_params.loc[aidx, f'{ww}_{vv}_{pp}'] = latent_fit['opt_hyper'][vv][widx]
                    fit_params.loc[aidx, f'AIC_{pp}'] = latent_fit['AIC']
                    fit_params.loc[aidx, f'BIC_{pp}'] = latent_fit['BIC']

                    # for psychometric plot
                    temp = pd.DataFrame()
                    temp['weighted_sum'] = latent_fit['weighted_sum']
                    temp['genotype'] = fit_params['genotype'][aidx]
                    temp['animal'] = animal
                    temp['choice'] = latent_fit['args']['dat']['inputs']['cBoth']
                    temp['gender'] = fit_params['gender'][aidx]
                    fit_psychometric[pp]= pd.concat(
                        (fit_psychometric[pp], temp))


                    model_label = 'Policy Gradient'
                    savefigpath = os.path.join(save_path, f'{animal}_{pp}_latent_fit_concat')
                    plot_latent_session(data, latent_fit, model_label,savefigpath)

        #%% plot summary figures

        #%% 1. plot fitted parameters. to do: make this work for both fit_mode
        
        if not fit_params.empty:
            metrics = [
                ('alpha', 'bias'),
                ('alpha', 'stim'),
                ('alpha', 'stick'),
                ('sigma', 'bias'),
                ('sigma', 'stim'),
                ('sigma', 'stick'),
            ]
            
            all_stats = []

            genders = fit_params['gender'].dropna().unique()
            genders = [g for g in genders if str(g).strip() != '']

            for protocol in protocols:
                if 'CD' in protocol and 'AB' not in protocol:
                    protocol = 'AB-'+protocol
                for gender in genders:
                    gender_df = fit_params[fit_params['gender'] == gender]
                    if fit_mode == 'session':
                        gender_df = gender_df[gender_df['protocol']==protocol]
                    genotype_groups = gender_df['genotype'].dropna().unique()
                    genotype_groups = [g for g in genotype_groups if str(g).strip() != '']
                    if {'WT', 'HET'}.issubset(set(genotype_groups)):
                        genotype_groups = ['WT', 'HET']
                    else:
                        genotype_groups = sorted(genotype_groups, key=lambda x: str(x))


                    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
                    stats_rows = []

                    color_map = {'WT': 'tab:blue', 'HET': 'tab:orange', 'KO': 'tab:green'}
                    legend_handles = []
                    for ax, (opt, weight) in zip(axes.flatten(), metrics):
                        if fit_mode == 'session':
                            col = f'{weight}_{opt}'
                        elif fit_mode == 'concat':
                            col = f'{weight}_{opt}_{protocol}'
                        plot_data = []
                        labels = []
                        for gt in genotype_groups:
                            if col in gender_df.columns:
                                values = pd.to_numeric(
                                    gender_df.loc[gender_df['genotype'] == gt, col],
                                    errors='coerce'
                                ).dropna().to_numpy()
                            else:
                                values = np.array([])
                            plot_data.append(values)
                            labels.append(gt)

                        bp = ax.boxplot(
                            plot_data,
                            labels=labels,
                            patch_artist=True,
                            showfliers=False,
                            medianprops={'color': 'black', 'linewidth': 1.5},
                            whiskerprops={'color': 'black'},
                            capprops={'color': 'black'},
                            flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4},
                        )
                        colors = [color_map.get(gt, 'lightgray') for gt in labels]
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_edgecolor('black')
                        for whisker in bp['whiskers']:
                            whisker.set_color('black')
                        for cap in bp['caps']:
                            cap.set_color('black')
                        for median in bp['medians']:
                            median.set_color('black')

                        #ax.set_facecolor('#fbfbfb')
                        #ax.grid(axis='y', color='gray', alpha=0.2, linestyle='--')

                        for x_pos, (gt, values) in enumerate(zip(labels, plot_data), start=1):
                            if values.size:
                                jitter = np.random.uniform(-0.08, 0.08, size=values.size)
                                ax.scatter(
                                    np.full(values.size, x_pos) + jitter,
                                    values,
                                    color=color_map.get(gt, 'black'),
                                    edgecolor='white',
                                    linewidth=0.5,
                                    s=40,
                                    alpha=0.9,
                                    zorder=3,
                                )
                        if len(labels) == 2 and labels[0] in color_map and labels[1] in color_map:
                            legend_handles = [Patch(facecolor=color_map[labels[0]], edgecolor='black', label=labels[0]),
                                                Patch(facecolor=color_map[labels[1]], edgecolor='black', label=labels[1])]

                        ax.set_title(f'{opt}_{weight}', fontsize=15)
                        ax.set_ylabel(opt, fontsize=10)
                        ax.tick_params(axis='both', labelsize=9)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        if len(genotype_groups) == 2 and col in gender_df.columns:
                            values_a = plot_data[0]
                            values_b = plot_data[1]
                            if values_a.size and values_b.size:
                                stat, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                            else:
                                stat, p_value = np.nan, np.nan
                            stats_rows.append({
                                'Protocol': protocol,
                                'Gender': gender,
                                'Opt': opt,
                                'Weight': weight,
                                'GroupA': genotype_groups[0],
                                'GroupB': genotype_groups[1],
                                'U': stat,
                                'PValue': p_value,
                            })

                    stats_df = pd.DataFrame(stats_rows)
                    if not stats_df.empty:
                        valid = stats_df['PValue'].notna()
                        if valid.any():
                            reject, adj_p, _, _ = multipletests(
                                stats_df.loc[valid, 'PValue'],
                                alpha=0.05,
                                method='fdr_bh'
                            )
                            stats_df.loc[valid, 'AdjustedPValue'] = adj_p
                            stats_df.loc[valid, 'Significant'] = reject
                        else:
                            stats_df['AdjustedPValue'] = np.nan
                            stats_df['Significant'] = False

                        all_stats.append(stats_df)

                        for ax, (opt, weight) in zip(axes.flatten(), metrics):
                            col = f'{weight}_{opt}_{protocol}'
                            subset = stats_df[(stats_df['Opt'] == opt) & (stats_df['Weight'] == weight)]
                            if subset.empty:
                                continue
                            annotations = []
                            for _, row in subset.iterrows():
                                if np.isfinite(row['AdjustedPValue']):
                                    annotations.append(
                                        f"FDR p={row['AdjustedPValue']:.3g}"
                                    )
                                else:
                                    annotations.append(
                                        f"FDR p=nan"
                                    )
                            annotation_text = '\n'.join(annotations)
                            ax.text(
                                0.5,
                                0.95,
                                annotation_text,
                                transform=ax.transAxes,
                                ha='center',
                                va='top',
                                fontsize=15,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
                            )

                    if legend_handles:
                        fig.legend(handles=legend_handles, loc='upper right', fontsize=10, frameon=False)
                    fig.subplots_adjust(top=0.88)
                    fig.tight_layout(rect=[0, 0, 1, 0.88])
                    fig.suptitle(f'{protocol} fitted params by gender {gender}', y=0.95, fontsize=20)
                    os.makedirs(f'{self.summary}/latent', exist_ok=True)
                    fig.savefig(
                        os.path.join(self.summary,'latent', f'{protocol}_fitted_params_by_gender_{gender}_{fit_mode}.png'),
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.2
                    )
                    fig.savefig(
                        os.path.join(self.summary,'latent', f'{protocol}_fitted_params_by_gender_{gender}_{fit_mode}.svg'),
                        bbox_inches='tight',
                        pad_inches=0.2
                    )
                    plt.close(fig)

            if all_stats:
                pd.concat(all_stats, ignore_index=True).to_csv(
                    os.path.join(self.summary, 'latent', 'fitted_params_mannwhitney.csv'),
                    index=False
                )

        # 2. plot psychometric curves
        # protocols = ['AB', 'CD']
        # all_stats = []

        # genders = fit_params['gender'].dropna().unique()
        # genders = [g for g in genders if str(g).strip() != '']
        # genotypes = fit_params['genotype'].dropna().unique()

        # max_weight = 5.95
        # min_weight = -5.95
        # step_weight = 0.1

        # for protocol in protocols:
        #     for gender in genders:
        #         # calculate binned weighted sum
        #         pR_data = pd.DataFrame() # calculate average right choice percentage for each animal
        #         pR_geno = []
        #         animals_plot = np.unique(self.data_index['Animal'][self.data_index['Gender'] == gender])
        #         bins = np.arange(
        #             min_weight - step_weight/2,
        #             max_weight + step_weight,
        #             step_weight
        #         )
        #         # calculate average right choice percentage for each animal
        #         for animal in animals_plot:
        #             pR_geno.append(self.Genotypes[np.array(self.Animals)==animal].values[0])
        #             #pR_data[animals] = np.full(len(bins), np.nan)
        #             weights = fit_psychometric[protocol].loc[
        #                     fit_psychometric[protocol]['animal'] == animal,
        #                     'weighted_sum'
        #                 ].to_numpy()
        #             choices = fit_psychometric[protocol].loc[
        #                     fit_psychometric[protocol]['animal'] == animal,
        #                     'choice'
        #                 ].to_numpy()
        #             choices_clean = np.array([x[0] for x in choices], dtype=float)
        #             temp = pd.DataFrame({
        #                 'weight': -weights,
        #                 'choice_right': np.array(choices_clean) + 0.5   # convert -0.5/0.5 to 0/1
        #             })

        #             # assign each trial to a bin
        #             temp['bin'] = pd.cut(
        #                 temp['weight'],
        #                 bins=bins,
        #                 labels=(bins[:-1] + bins[1:]) / 2
        #             )

        #             # mean choice in each bin = P(right)
        #             p_right = (
        #                     temp.groupby('bin', observed=True)['choice_right']
        #                     .mean()
        #                     .reindex(
        #                         (bins[:-1] + bins[1:]) / 2
        #                     )
        #                 )

        #             # bin centers and corresponding percentages
        #             bin_centers = p_right.index.astype(float)
        #             p_right = p_right.to_numpy()
        #             pR_data[animal] = p_right

        #         plotMask = fit_psychometric[protocol]['gender'] == gender
        #         temp = fit_psychometric[protocol].loc[plotMask].copy()

        #         temp['Weight'] = pd.cut(
        #             temp['weighted_sum'],
        #             bins=bins,
        #             labels=np.arange(min_weight, max_weight + step_weight, step_weight)
        #         )

        #         binned_weight_count = (
        #             temp.groupby(['Weight', 'genotype'])
        #                 .size()
        #                 .unstack(fill_value=0)
        #                 .reset_index()
        #         )

        #         # calculate average reward to choose right 
        #         # calculate average reward to choose right
        #         fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        #         for gidx, geno in enumerate(genotypes):
        #             nGeno = np.sum(np.array(pR_geno) == geno)

        #             # histogram
        #             ax[gidx].bar(
        #                 binned_weight_count['Weight'],
        #                 binned_weight_count[geno],
        #                 width=step_weight,
        #                 color='black',
        #                 align='center'
        #             )

        #             # remove top and right spines of main axis
        #             ax[gidx].spines['top'].set_visible(False)
        #             ax[gidx].spines['right'].set_visible(False)

        #             # left y-label only on left plot
        #             if gidx == 0:
        #                 ax[gidx].set_ylabel('Count')

        #             # second y-axis
        #             ax2 = ax[gidx].twinx()

        #             mean_pR = np.nanmean(
        #                 pR_data.loc[:, np.array(pR_geno) == geno],
        #                 axis=1
        #             )
        #             ste_pR = (
        #                 np.nanstd(
        #                     pR_data.loc[:, np.array(pR_geno) == geno],
        #                     axis=1
        #                 ) / np.sqrt(nGeno)
        #             )

        #             ax2.errorbar(
        #                 bin_centers,
        #                 mean_pR,
        #                 yerr=ste_pR,
        #                 fmt='o',
        #                 color='C0'
        #             )

        #             # theoretical probability (softmax)
        #             ax2.plot(bin_centers, expit(bin_centers), 'r-')

        #             # remove top spine of right axis
        #             ax2.spines['top'].set_visible(False)

        #             # show right y-label only on right plot
        #             if gidx == 1:
        #                 ax2.set_ylabel('P(choice right)')
        #             else:
        #                 ax2.set_yticklabels([])

        #             ax[gidx].set_xlabel('Weighted sum')
        #             ax[gidx].set_title(geno)

        #         fig.subplots_adjust(top=0.88)
        #         fig.tight_layout(rect=[0, 0, 1, 0.88])
        #         fig.suptitle(f'{protocol} psychometric curve {gender}', y=0.95, fontsize=20)
        #         os.makedirs(f'{self.summary}/latent', exist_ok=True)
        #         fig.savefig(
        #             os.path.join(self.summary,'latent', f'{protocol}_psychometric_curve_by_gender_{gender}_{fit_mode}.png'),
        #             dpi=300,
        #             bbox_inches='tight',
        #             pad_inches=0.2
        #         )
        #         fig.savefig(
        #             os.path.join(self.summary,'latent', f'{protocol}_psychometric_curve_by_gender_{gender}_{fit_mode}.svg'),
        #             bbox_inches='tight',
        #             pad_inches=0.2
        #         )
        #         plt.close(fig)

    def model_comparison(self):
        # compare AIC result for hybrid model and policy gradient model
        
        session_to_compare = ['AB1', 'AB2', 'AB3', 'CD1', 'CD2', 'CD3']
        mAIC_hybrid = {}
        mAIC_policy = {}
        delta_AIC = {} # policy - hybrid
        subject_ID = {}
        genotypes = {}

        genders = ['M', 'F']
        for gender in genders:
            genderIDs = np.array(self.Animals)[np.array(self.Gender) == gender]

            for ses in session_to_compare:
                # load hybrid model results
                from scipy.io import loadmat
                hybrid_name = os.path.join(self.summary, 'Results', 'hybrid_fit_' + ses + '.mat')
                mat = loadmat(hybrid_name)
                fit_result = mat['fit_result']
                subjects = fit_result['subjects'][0][0]
                tempSub = []
                tempAIC = []
                tempGeno = []
                for sub in subjects:
                    if str(sub[0]) in genderIDs:
                        tempSub.append(sub)
                        tempAIC.append(fit_result['All_fits'][0][0][np.where(fit_result['subjects'][0][0] == sub)[0][0], 2])
                        tempGeno.append(fit_result['genotypes'][0][0][np.where(fit_result['subjects'][0][0] == sub)[0][0]])

                subject_ID[ses] = tempSub
                genotypes[ses] = tempGeno
                AIC_hybrid = np.array(tempAIC)
                #all_params = fit_result['All_params']

                # load policy gradient model results
                if 'AB' in ses:
                    target_ses = ses[0:2]
                elif 'CD' in ses:
                    target_ses = 'AB-' + ses[0:2]


                AIC_policy = []
                subject_policy = []
                for sub in tempSub:
                    
                    animal_mask = self.data_index['Animal'] == str(sub[0])
                    protocol_mask = self.data_index['Protocol'] == target_ses
                    day_mask = self.data_index['ProtocolDay'] == int(ses[2])
                    match_idx = self.data_index.index[animal_mask & protocol_mask & day_mask]

                    if len(match_idx) == 0:
                        continue
                    subject_policy.append(str(sub[0]))
                    ss = match_idx[0]
                    save_path = os.path.join(self.data_index.loc[ss, 'AnalysisPath'], 'latent')

                    savedatapath = os.path.join(save_path, 'policy_gradient_fit.json')
                    # load json file
                    with open(savedatapath, 'r') as f:
                        policy_gradient_fit = json.load(f)
                        AIC_policy.append(policy_gradient_fit['AIC'])
                        # animal_PG.append(self.data_index['Animal'][ss])
                mAIC_policy[ses] = np.array(AIC_policy) - (np.array(AIC_policy) + np.array(AIC_hybrid))/2
                mAIC_hybrid[ses] = np.array(AIC_hybrid) - (np.array(AIC_hybrid) + np.array(AIC_policy))/2

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(session_to_compare))
            offset = 0.18
            colors = {'Policy gradient': 'C0', 'Hybrid': 'C1'}

            for idx, ses in enumerate(session_to_compare):
                policy_values = mAIC_policy[ses][np.isfinite(mAIC_policy[ses])]
                hybrid_values = mAIC_hybrid[ses][np.isfinite(mAIC_hybrid[ses])]

                for values, label, position in (
                    (policy_values, 'Policy gradient', x[idx] - offset),
                    (hybrid_values, 'Hybrid', x[idx] + offset),
                ):
                    if values.size:
                        box = ax.boxplot(
                            values, positions=[position], widths=0.3, patch_artist=True,
                            showfliers=False,
                        )
                        for element in ('boxes', 'whiskers', 'caps', 'medians'):
                            plt.setp(box[element], color=colors[label])
                        box['boxes'][0].set_facecolor(colors[label])
                        box['boxes'][0].set_alpha(0.35)
                        jitter = np.linspace(-0.06, 0.06, values.size) if values.size > 1 else [0]
                        ax.scatter(
                            position + jitter, values, color=colors[label], s=25,
                            alpha=0.8, zorder=3,
                            label=label if idx == 0 else None,
                        )

                if policy_values.size and hybrid_values.size:
                    _, p_value = wilcoxon(policy_values, hybrid_values, alternative='two-sided')
                else:
                    p_value = np.nan
                y_max = max(
                    np.max(policy_values) if policy_values.size else -np.inf,
                    np.max(hybrid_values) if hybrid_values.size else -np.inf,
                )
                if np.isfinite(y_max):
                    ax.annotate(
                        f'p = {p_value:.3g}' if np.isfinite(p_value) else 'p = n/a',
                        (x[idx], y_max), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                    )

            raw_p_values = []
            for ses in session_to_compare:
                policy_values = mAIC_policy[ses][np.isfinite(mAIC_policy[ses])]
                hybrid_values = mAIC_hybrid[ses][np.isfinite(mAIC_hybrid[ses])]
                raw_p_values.append(
                wilcoxon(policy_values, hybrid_values, alternative='two-sided').pvalue
                    if policy_values.size and hybrid_values.size else np.nan
                )
            valid_p_values = np.isfinite(raw_p_values)
            adjusted_p_values = np.full(len(raw_p_values), np.nan)
            if np.any(valid_p_values):
                adjusted_p_values[valid_p_values] = multipletests(
                    np.asarray(raw_p_values)[valid_p_values], method='fdr_bh'
                )[1]

            annotation_idx = 0
            for ses, adjusted_p_value in zip(session_to_compare, adjusted_p_values):
                values = np.concatenate((mAIC_policy[ses], mAIC_hybrid[ses]))
                if np.any(np.isfinite(values)):
                    ax.texts[annotation_idx].set_text(
                        f'FDR p = {adjusted_p_value:.3g}'
                        if np.isfinite(adjusted_p_value) else 'FDR p = n/a'
                    )
                    ax.texts[annotation_idx].set_fontsize(12)
                    annotation_idx += 1

            all_values = np.concatenate([
                np.concatenate((mAIC_policy[ses], mAIC_hybrid[ses]))
                for ses in session_to_compare
            ])
            finite_values = all_values[np.isfinite(all_values)]
            if finite_values.size:
                y_lower, y_upper = np.percentile(finite_values, [1, 99])
                padding = max((y_upper - y_lower) * 0.08, 0.1)
                ax.set_ylim(y_lower - padding, y_upper + padding)
                for annotation in ax.texts:
                    annotation.xy = (annotation.xy[0], y_upper - padding)

            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xticks(x, session_to_compare)
            ax.set_xlabel('Session')
            ax.set_ylabel('Mean-centered AIC')
            ax.set_title(f'Policy gradient and hybrid model comparison {self.strain} {gender}')
            ax.legend(
                handles=[
                    Patch(facecolor=colors['Policy gradient'], edgecolor=colors['Policy gradient'],
                        alpha=0.35, label='Policy gradient'),
                    Patch(facecolor=colors['Hybrid'], edgecolor=colors['Hybrid'],
                        alpha=0.35, label='Hybrid'),
                ],
                frameon=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            os.makedirs(os.path.join(self.summary, 'latent'), exist_ok=True)
            fig.savefig(os.path.join(self.summary, 'latent', f'model_comparison_mAIC_{self.strain}_{gender}.png'), dpi=300)
            fig.savefig(os.path.join(self.summary, 'latent', f'model_comparison_mAIC_{self.strain}_{gender}.svg'))
            #plt.close(fig)
        
    def odor_summary(self):
        # plot summary figures for model fitting
        # 1. fitted parameters (genotype comparisons)
        # 2. psychometric curves
        # 3. weights of bias and stickiness pre/post learning
        pass

    def find_eureka(self):
        """ for each animal, concatenate the first 3 AB sessions, makes a decision on when learning occurs
        then look for a peak of the derivative of the fitted weights"""

        # for each animal, concatenate AB sessions before the first AB-CD session
        # and concatenate the 3 AB-CD sessions
        saveData = {}
        learning_summary_rows = []
        for animal in self.data_index['Animal'].unique():
            saveData[animal] = {}
            saveData[animal]['AB'] = {}
            saveData[animal]['CD'] = {}
            result_concat = {}
            result_concat['AB'] = pd.DataFrame()
            result_concat['CD'] = pd.DataFrame()
            pgFit = {}
            pgFit['AB'] = []
            pgFit['CD'] = []
            Animal_sessions = self.data_index[self.data_index['Animal'] == animal]
            AB_CD_1_idx = Animal_sessions[Animal_sessions['Protocol'].str.contains('AB-CD')].index[0] if not Animal_sessions[Animal_sessions['Protocol'].str.contains('AB-CD')].empty else None            
            if AB_CD_1_idx is not None:
                AB_sessions = Animal_sessions.loc[:AB_CD_1_idx-1]
            else:
                AB_sessions = Animal_sessions
            AB_CD_sessions = Animal_sessions[np.logical_and(Animal_sessions['Protocol']=='AB-CD', 
                                                            Animal_sessions['ProtocolDay'] <=3 )]

            for sIdx in AB_sessions.index:
                temp_result = pd.read_csv(AB_sessions.loc[sIdx]['BehCSV'])
                # remove miss trials
                temp_result = temp_result[~np.isnan(temp_result['actions'])]
                result_concat['AB'] = pd.concat([result_concat['AB'], temp_result], ignore_index=True)

                # load policy gradient fit
                savedatapath = os.path.join(AB_sessions.loc[sIdx]['AnalysisPath'], 'latent', 'policy_gradient_fit.json')
                if os.path.exists(savedatapath):
                    with open(savedatapath, 'r') as f:
                        fitResult = json.load(f)
                        fitted_weights = np.array(fitResult['wMode'])
                    pgFit['AB'].append(fitted_weights)
            if len(pgFit['AB']) > 0:
                pgFit['AB'] = np.concatenate(pgFit['AB'], axis=1)

            for sIdx in AB_CD_sessions.index:
                temp_result = pd.read_csv(AB_CD_sessions.loc[sIdx]['BehCSV'])
                # remove miss trials
                temp_result = temp_result[~np.isnan(temp_result['actions'])]
                # remove AB trials
                temp_result = temp_result[temp_result['schedule']>2]
                result_concat['CD'] = pd.concat([result_concat['CD'], temp_result], ignore_index=True)
                
                # load policy gradient fit
                savedatapath = os.path.join(AB_CD_sessions.loc[sIdx]['AnalysisPath'], 'latent', 'policy_gradient_fit.json')
                if os.path.exists(savedatapath):
                    with open(savedatapath, 'r') as f:
                        fitResult = json.load(f)
                        fitted_weights = np.array(fitResult['wMode'])
                    pgFit['CD'].append(fitted_weights)
            if len(pgFit['CD']) > 0:
                pgFit['CD'] = np.concatenate(pgFit['CD'], axis=1)
            
            # calculate running reward rate
            protocols = ['AB', 'CD']
            for pp in protocols:
                if result_concat[pp].empty:
                    reward = np.array([np.nan])
                else:
                    reward = result_concat[pp]['reward'].fillna(0)
                    reward = reward.replace([2, 3], 1)
                rewarded = (reward > 0).astype(float)
                n_trials = len(rewarded)
                window_size = 60
                running_reward_prob = np.full(n_trials, np.nan)
            
                if n_trials >= window_size:
                    csum = np.empty(n_trials + 1, dtype=float)
                    csum[0] = 0.0
                    np.cumsum(rewarded.to_numpy(), out=csum[1:])
                    running_reward_prob[:n_trials - window_size + 1] = (
                        csum[window_size:] - csum[:-window_size]
                    ) / window_size
                pCorrect = pd.Series(running_reward_prob).rolling(500, center=True, min_periods=1).mean().to_numpy()
            
                p = np.asarray(pCorrect, dtype=float)
                valid_idx = np.flatnonzero(np.isfinite(p))
                learned_level = 0.6
                plateau_stability_tol = 0.03
                # choose plateau window length relative to the available data
                # so it adapts for long or short sessions rather than using a hard 200-trial window.
                plateau_min_frac = 0.05
                plateau_min_n = max(100, int(np.ceil(valid_idx.size * plateau_min_frac)))
                plateau_min_n = min(plateau_min_n, 250)
                learning_window = (np.nan, np.nan)
                learning_window_perf = (np.nan, np.nan)
                plateau_window = None
                plateau_perf = np.nan
                sigmoid_tau = np.nan
                sigmoid_boundary_10_90 = (np.nan, np.nan)
                sigmoid_params = (np.nan, np.nan, np.nan, np.nan)
                x_curve = valid_idx.astype(float) + 1
                y_curve = []
                if valid_idx.size:
                    p_valid = p[valid_idx]
                    above_learned = p_valid >= learned_level
                    # identify windows of length `plateau_min_n` where the
                    # performance is stable (max-min <= tolerance) and the
                    # window median is above the learned level. This finds the
                    # start of a sustained, stable plateau rather than the
                    # first time the curve crosses the threshold.
                    plateau_start_pos = None
                    if p_valid.size >= plateau_min_n:
                        try:
                            from numpy.lib.stride_tricks import sliding_window_view
                            windows = sliding_window_view(p_valid, plateau_min_n)
                        except Exception:
                            # fallback to manual stacking if sliding_window_view
                            # isn't available
                            windows = np.vstack([
                                p_valid[i: i + plateau_min_n]
                                for i in range(p_valid.size - plateau_min_n + 1)
                            ])

                        # window median and range
                        window_median = np.nanmedian(windows, axis=1)
                        window_range = np.nanmax(windows, axis=1) - np.nanmin(windows, axis=1)

                        candidates = np.flatnonzero((window_median >= learned_level) & (window_range <= plateau_stability_tol))
                        if candidates.size:
                            plateau_start_pos = int(candidates[0])
                    if plateau_start_pos is not None:
                        plateau_end_pos = plateau_start_pos + plateau_min_n - 1
                        while (
                            plateau_end_pos + 1 < above_learned.size
                            and above_learned[plateau_end_pos + 1]
                        ):
                            # Check if expanding the window maintains stability:
                            # (1) range is small, and (2) no sustained trend (first half median ≈ second half median)
                            window = p_valid[plateau_end_pos - plateau_min_n + 2:plateau_end_pos + 2]
                            if window.size > 0:
                                window_range = np.nanmax(window) - np.nanmin(window)
                                # Split window into halves and check medians (detect sustained rise/fall)
                                mid = len(window) // 2
                                first_half_median = np.nanmedian(window[:mid]) if mid > 0 else np.nan
                                second_half_median = np.nanmedian(window[mid:]) if (len(window) - mid) > 0 else np.nan
                                trend_magnitude = abs(second_half_median - first_half_median)
                                
                                # Only expand if range is small AND trend is negligible
                                if window_range <= plateau_stability_tol and trend_magnitude <= plateau_stability_tol / 2:
                                    plateau_end_pos += 1
                                else:
                                    break
                            else:
                                break
                        plateau_window = (
                            int(valid_idx[plateau_start_pos] + 1),
                            int(valid_idx[plateau_end_pos] + 1)
                        )
                        plateau_perf = float(np.nanmedian(p_valid[plateau_start_pos:plateau_end_pos + 1]))

                        # refine plateau start to the first point where performance
                        # reaches (or nearly reaches) the identified plateau performance.
                        # This moves the reported start to the point where the curve
                        # actually attains the plateau level (useful for cases like
                        # animal '381' where plateau_perf ~ 0.8 and the true start
                        # is later than the learned_level crossing).
                        try:
                            plateau_thresh = plateau_perf - 0.02
                            if plateau_thresh < learned_level:
                                plateau_thresh = learned_level

                            # find earliest index in p_valid that reaches threshold
                            reach_idxs = np.flatnonzero(p_valid >= plateau_thresh)
                            if reach_idxs.size:
                                # pick the first occurrence that is not after the originally
                                # detected plateau start (guard against odd cases)
                                new_start_pos = int(reach_idxs[0])
                                if new_start_pos > plateau_start_pos:
                                    plateau_start_pos = new_start_pos
                                    # update plateau_window start coordinate
                                    plateau_window = (
                                        int(valid_idx[plateau_start_pos] + 1),
                                        plateau_window[1]
                                    )
                        except Exception:
                            # if anything goes wrong, keep original plateau_start_pos
                            pass

                        start_search = p_valid[:plateau_start_pos + 1]
                        if start_search.size:
                            baseline_n = max(1, min(500, start_search.size // 2))
                            baseline_level = np.nanmedian(start_search[:baseline_n])
                            rise_threshold = max(0.02, 0.2 * (np.nanmedian(p_valid[plateau_start_pos:plateau_end_pos + 1]) - baseline_level))
                            transition_mask = start_search >= baseline_level + rise_threshold
                            transition_n = min(10, transition_mask.size)
                            sustained_rise = np.convolve(
                                transition_mask.astype(int),
                                np.ones(transition_n, dtype=int),
                                mode='valid'
                            ) == transition_n
                            rise_pos = np.flatnonzero(sustained_rise)
                            if rise_pos.size:
                                chance_pos = np.flatnonzero(start_search[:rise_pos[0] + 1] <= baseline_level + 0.02)
                                start_pos = chance_pos[-1] if chance_pos.size else rise_pos[0]
                            else:
                                start_pos = 0
                        else:
                            start_pos = 0

                        start_idx = valid_idx[start_pos]
                        end_idx = valid_idx[plateau_start_pos]
                        learning_window = (int(start_idx + 1), int(end_idx + 1))
                        learning_window_perf = (float(p[start_idx]), float(p[end_idx]))
                        fit_pad = max(250, end_idx - start_idx)
                        fit_start = max(0, start_idx - fit_pad)
                        fit_end_idx = min(
                            valid_idx[plateau_start_pos] + 400,
                            valid_idx[plateau_end_pos]
                        )
                        fit_end = min(len(p), fit_end_idx + 1)
                        fit_idx = np.arange(fit_start, fit_end)
                        fit_idx = fit_idx[np.isfinite(p[fit_idx])]
                        if fit_idx.size >= 8:
                            x_fit = fit_idx.astype(float) + 1
                            y_fit = p[fit_idx]

                            # fix the high asymptote to the estimated plateau performance
                            def sigmoid(params, x):
                                p_low, k, tau = params
                                p_delta = plateau_perf - p_low
                                return p_low + p_delta / (1 + np.exp(-k * (x - tau)))

                            def sigmoid_loss(params):
                                y_hat = sigmoid(params, x_fit)
                                return np.mean((y_fit - y_hat) ** 2)

                            p_low0 = float(np.nanpercentile(y_fit, 10))
                            p_low0 = min(p_low0, plateau_perf - 0.02)
                            initial = [
                                p_low0,
                                0.01,
                                float((learning_window[0] + learning_window[1]) / 2),
                            ]
                            bounds = [
                                (0.0, plateau_perf - 1e-6),
                                (1e-6, None),
                                (float(x_fit[0]), float(x_fit[-1])),
                            ]
                            result = minimize(sigmoid_loss, initial, bounds=bounds, method='L-BFGS-B')
                            if result.success:
                                p_low, k, tau = result.x
                                p_delta = plateau_perf - p_low
                                sigmoid_tau = float(tau)
                                sigmoid_params = (float(p_low), float(p_delta), float(k), float(tau))
                                delta_t = np.log(9) / k
                                sigmoid_boundary_10_90 = (float(tau - delta_t), float(tau + delta_t))
                                
                                y_curve = sigmoid(result.x, x_curve)

                    else:
                        # no learning occurred, set the tau to be the total number of trials
                        sigmoid_boundary_10_90 = [np.nan, np.nan]
                        sigmoid_tau = np.nan

                    if plateau_window is None:
                        saveData[animal][pp]['learning_window'] = None
                        saveData[animal][pp]['learning_window_perf'] = None
                        saveData[animal][pp]['plateau_window'] = None
                        saveData[animal][pp]['plateau_perf'] = None
                        saveData[animal][pp]['sigmoid_tau'] = len(valid_idx)
                        saveData[animal][pp]['sigmoid_boundary_10_90'] = None
                        saveData[animal][pp]['sigmoid_params'] = None
                        saveData[animal][pp]['sigmoid_par_name'] = None
                    else:
                        saveData[animal][pp]['learning_window'] = learning_window
                        saveData[animal][pp]['learning_window_perf'] = learning_window_perf
                        saveData[animal][pp]['plateau_window'] = plateau_window
                        saveData[animal][pp]['plateau_perf'] = plateau_perf
                        saveData[animal][pp]['sigmoid_tau'] = sigmoid_tau
                        saveData[animal][pp]['sigmoid_boundary_10_90'] = sigmoid_boundary_10_90
                        saveData[animal][pp]['sigmoid_params'] = sigmoid_params
                        saveData[animal][pp]['sigmoid_par_name'] = ['p_low', 'p_delta', 'k', 'tau']
                        startTrial = int(sigmoid_boundary_10_90[0])
                        endTrial = int(sigmoid_boundary_10_90[1])
                        saveData[animal][pp]['response_times_preLearning'] = result_concat[pp]['outcome'][0:startTrial] - result_concat[pp]['center_in'][0:startTrial]
                        saveData[animal][pp]['response_times_duringLearning'] = result_concat[pp]['outcome'][startTrial:endTrial] - result_concat[pp]['center_in'][startTrial:endTrial]
                        saveData[animal][pp]['intertrial_intervals_preLearning'] = np.diff(result_concat[pp]['center_in'][0:startTrial])
                        saveData[animal][pp]['intertrial_intervals_duringLearning'] = np.diff(result_concat[pp]['center_in'][startTrial:endTrial])
                    # get the response times and intertrial intervals before learning
                    # and during learning

                    learning_summary_rows.append({
                        'Animal': animal,
                        'Gender': Animal_sessions['Gender'].iloc[0],
                        'Genotype': Animal_sessions['Genotype'].iloc[0],
                        'Protocol': pp,
                        'FittedTau': saveData[animal][pp]['sigmoid_tau'],
                        'PlateauPerformance': saveData[animal][pp]['plateau_perf'],
                        'FittedK': (
                            saveData[animal][pp]['sigmoid_params'][2]
                            if saveData[animal][pp]['sigmoid_params'] is not None
                            else np.nan
                        ),
                    })

                    # make the plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(x_curve, p[valid_idx], color='black', linewidth=2, label='Data')
                    # plot y = 0.5 line
                    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5)
                    if plateau_window is not None:
                        ax.axvspan(
                            plateau_window[0],
                            plateau_window[1],
                            color='green',
                            alpha=0.12,
                            label='First plateau'
                        )
                    if len(y_curve) > 0:
                        ax.plot(x_curve, y_curve, color='red', linewidth=2, label='Sigmoid fit')
                        ax.axvspan(
                            sigmoid_boundary_10_90[0],
                            sigmoid_boundary_10_90[1],
                            color='red',
                            alpha=0.15,
                            label='10-90% boundary'
                        )
                        ax.axvline(sigmoid_tau, color='red', linestyle='--', linewidth=1.5, label='Tau')
                    ax.set_xlabel('Trial')
                    ax.set_ylabel('P(correct)')
                    ax.set_ylim(0, 1)
                    ax.set_title(f'{animal} {pp} learning curve')
                    ax.legend(frameon=False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.tight_layout()
                    os.makedirs(os.path.join(self.summary, animal), exist_ok=True)
                    fig.savefig(
                        os.path.join(self.summary,animal, f'{animal}_{pp}_learning_sigmoid.png'),
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close(fig)
                    
            if not hasattr(self, 'eureka_learning'):
                self.eureka_learning = {}
            self.eureka_learning[animal] = saveData[animal]

        eureka_learning_summary = pd.DataFrame.from_records(
            learning_summary_rows,
            columns=['Animal', 'Genotype', 'Gender','Protocol', 'FittedTau', 'PlateauPerformance', 'FittedK']
        )
        os.makedirs(self.summary, exist_ok=True)
        eureka_learning_summary.to_csv(os.path.join(self.summary, 'eureka_learning_summary.csv'), index=False)
        self.eureka_learning_summary = eureka_learning_summary

        for gender in eureka_learning_summary['Gender'].unique():
            gender_df = eureka_learning_summary[eureka_learning_summary['Gender'] == gender]
            metrics = [
                ('FittedTau', 'Fitted tau'),
                ('FittedK', 'Fitted K'),
            ]
            genotype_order = [g for g in ['WT', 'HET', 'KO'] if g in set(gender_df['Genotype'].dropna())]
            genotype_order += [g for g in gender_df['Genotype'].dropna().unique() if g not in genotype_order]
            stats_rows = []

            for protocol, protocol_df in gender_df.groupby('Protocol', sort=False):
                if protocol_df.empty or not genotype_order:
                    continue

                for metric, _ in metrics:
                    plot_data = [
                        pd.to_numeric(
                            protocol_df.loc[protocol_df['Genotype'] == genotype, metric],
                            errors='coerce'
                        ).dropna().to_numpy()
                        for genotype in genotype_order
                    ]
                    for i, genotype_a in enumerate(genotype_order):
                        values_a = plot_data[i]
                        for j in range(i + 1, len(genotype_order)):
                            genotype_b = genotype_order[j]
                            values_b = plot_data[j]
                            if values_a.size and values_b.size:
                                stat, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                            else:
                                stat, p_value = np.nan, np.nan
                            stats_rows.append({
                                'Protocol': protocol,
                                'Metric': metric,
                                'GenotypeA': genotype_a,
                                'GenotypeB': genotype_b,
                                'N_A': values_a.size,
                                'N_B': values_b.size,
                                'U': stat,
                                'PValue': p_value,
                            })

            eureka_learning_stats = pd.DataFrame(
                stats_rows,
                columns=['Protocol', 'Metric', 'GenotypeA', 'GenotypeB', 'N_A', 'N_B', 'U', 'PValue']
            )
            eureka_learning_stats['AdjustedPValue'] = np.nan
            eureka_learning_stats['Significant'] = False
            eureka_learning_stats['Gender'] = gender
            valid_p = np.isfinite(eureka_learning_stats['PValue'])
            if valid_p.any():
                reject, adjusted_p, _, _ = multipletests(
                    eureka_learning_stats.loc[valid_p, 'PValue'],
                    alpha=0.05,
                    method='fdr_bh'
                )
                eureka_learning_stats.loc[valid_p, 'AdjustedPValue'] = adjusted_p
                eureka_learning_stats.loc[valid_p, 'Significant'] = reject

            for protocol, protocol_df in gender_df.groupby('Protocol', sort=False):
                if protocol_df.empty or not genotype_order:
                    continue

                fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
                axes = axes[0]
                for ax, (metric, ylabel) in zip(axes, metrics):
                    plot_data = [
                        pd.to_numeric(
                            protocol_df.loc[protocol_df['Genotype'] == genotype, metric],
                            errors='coerce'
                        ).dropna().to_numpy()
                        for genotype in genotype_order
                    ]
                    ax.boxplot(
                        plot_data,
                        labels=genotype_order,
                        patch_artist=True,
                        showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        boxprops={'facecolor': 'white', 'edgecolor': 'black'},
                        whiskerprops={'color': 'black'},
                        capprops={'color': 'black'},
                    )
                    for x_pos, values in enumerate(plot_data, start=1):
                        if values.size:
                            jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
                            ax.scatter(
                                np.full(values.size, x_pos) + jitter,
                                values,
                                color='black',
                                s=25,
                                alpha=0.8,
                                zorder=3,
                            )
                    metric_stats = eureka_learning_stats[
                        (eureka_learning_stats['Protocol'] == protocol) &
                        (eureka_learning_stats['Metric'] == metric)
                    ]
                    y_values = np.concatenate([values for values in plot_data if values.size]) if any(values.size for values in plot_data) else np.array([])
                    if y_values.size and not metric_stats.empty:
                        y_min = float(np.nanmin(y_values))
                        y_max = float(np.nanmax(y_values))
                        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
                        y_base = y_max + 0.08 * y_span
                        y_step = 0.12 * y_span
                        for row_idx, (_, row) in enumerate(metric_stats.iterrows()):
                            x1 = genotype_order.index(row['GenotypeA']) + 1
                            x2 = genotype_order.index(row['GenotypeB']) + 1
                            y = y_base + row_idx * y_step
                            y_bracket = y + 0.02 * y_span
                            ax.plot([x1, x1, x2, x2], [y, y_bracket, y_bracket, y], color='black', linewidth=1)
                            adj_p = row['AdjustedPValue']
                            p_label = f"FDR p={adj_p:.3g}" if np.isfinite(adj_p) else "FDR p=nan"
                            ax.text((x1 + x2) / 2, y_bracket + 0.005 * y_span, p_label, ha='center', va='bottom', fontsize=8)
                        ax.set_ylim(top=y_base + len(metric_stats) * y_step + 0.08 * y_span)
                    ax.set_title(ylabel)
                    ax.set_ylabel(ylabel)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                fig.suptitle(f'{protocol} learning fit by genotype {gender}')
                fig.tight_layout()
                fig.savefig(
                    os.path.join(self.summary, f'{protocol}_learning_fit_by_genotype_{gender}.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close(fig)

        eureka_learning_stats.to_csv(os.path.join(self.summary, 'eureka_learning_mannwhitney.csv'), index=False)
        self.eureka_learning_stats = eureka_learning_stats

    def plot_response_times(self):
        """Plot distributions of response times and intertrial intervals by genotype."""
        if not hasattr(self, 'eureka_learning'):
            print("eureka_learning not found. Run find_eureka first.")
            return
        
        # Collect data by genotype and period
        data_by_genotype = {}
        
        for animal in self.eureka_learning:
            genotype = self.data_index[self.data_index['Animal'] == animal]['Genotype'].iloc[0]
            
            if genotype not in data_by_genotype:
                data_by_genotype[genotype] = {
                    'response_times_preLearning': [],
                    'response_times_duringLearning': [],
                    'intertrial_intervals_preLearning': [],
                    'intertrial_intervals_duringLearning': [],
                }
            
            for protocol in self.eureka_learning[animal]:
                if isinstance(self.eureka_learning[animal][protocol], dict):
                    data = self.eureka_learning[animal][protocol]
                    
                    # Collect response times
                    if 'response_times_preLearning' in data :
                        data_by_genotype[genotype]['response_times_preLearning'].extend(data['response_times_preLearning'])
                    if 'response_times_duringLearning' in data :
                        data_by_genotype[genotype]['response_times_duringLearning'].extend(data['response_times_duringLearning'])
                    
                    # Collect intertrial intervals
                    if 'intertrial_intervals_preLearning' in data :
                        data_by_genotype[genotype]['intertrial_intervals_preLearning'].extend(data['intertrial_intervals_preLearning'])
                    if 'intertrial_intervals_duringLearning' in data :
                        data_by_genotype[genotype]['intertrial_intervals_duringLearning'].extend(data['intertrial_intervals_duringLearning'])
        
        if not data_by_genotype:
            print("No timing data found in eureka_learning")
            return
        
        genotypes = sorted(data_by_genotype.keys())
        colors = {'WT': 'black', 'HET': 'orange', 'KO': 'red'}
        
        # Plot response times
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pre-learning
        ax = axes[0]
        for genotype in genotypes:
            rt_pre = np.array(data_by_genotype[genotype]['response_times_preLearning'])
            rt_pre = rt_pre[np.isfinite(rt_pre)]
            if rt_pre.size > 0:
                ax.hist(rt_pre, bins=30, alpha=0.6, label=genotype, color=colors.get(genotype, 'gray'))
        ax.set_xlabel('Response time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Times (Pre-Learning)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # During learning
        ax = axes[1]
        for genotype in genotypes:
            rt_during = np.array(data_by_genotype[genotype]['response_times_duringLearning'])
            rt_during = rt_during[np.isfinite(rt_during)]
            if rt_during.size > 0:
                ax.hist(rt_during, bins=30, alpha=0.6, label=genotype, color=colors.get(genotype, 'gray'))
        ax.set_xlabel('Response time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Times (During Learning)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.summary, 'response_times_by_genotype.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Plot intertrial intervals
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pre-learning
        ax = axes[0]
        for genotype in genotypes:
            iti_pre = np.array(data_by_genotype[genotype]['intertrial_intervals_preLearning'])
            iti_pre = iti_pre[np.isfinite(iti_pre)]
            if iti_pre.size > 0:
                ax.hist(iti_pre, bins=30, alpha=0.6, label=genotype, color=colors.get(genotype, 'gray'))
        ax.set_xlabel('Intertrial interval (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Intertrial Intervals (Pre-Learning)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # During learning
        ax = axes[1]
        for genotype in genotypes:
            iti_during = np.array(data_by_genotype[genotype]['intertrial_intervals_duringLearning'])
            iti_during = iti_during[np.isfinite(iti_during)]
            if iti_during.size > 0:
                ax.hist(iti_during, bins=30, alpha=0.6, label=genotype, color=colors.get(genotype, 'gray'))
        ax.set_xlabel('Intertrial interval (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Intertrial Intervals (During Learning)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.summary, 'intertrial_intervals_by_genotype.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Response times and intertrial intervals plots saved to {self.summary}")

    def plot_performance(self):
        # call matlab function to plot the performance

        perf_df = pd.DataFrame(columns=['Animal','Gender', 'Genotype', 'Date', 'Protocol', 'ProtocolDay', 'RewardRate', 'd'])
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
            perf_df.loc[bIdx, 'Gender'] = self.data_index['Gender'][bIdx]

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
        #%% determine sex
        Sexes = perf_df['Gender'].unique()

        perf_plot_AB1 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_AB2 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_AB3 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD1 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD2 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        perf_plot_CD3 = pd.DataFrame(columns=['Animal', 'Gender', 'Genotype', 'Block', 'RewardRate', 'd'])
        for idx, row in perf_df.iterrows():
            for bb in range(10):
                if not np.isnan(row['RewardRate'][bb]):
                    if row['Protocol'] == 'AB':
                        if row['ProtocolDay'] == 1:   
                            perf_plot_AB1 = pd.concat([perf_plot_AB1, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 2:
                            perf_plot_AB2 = pd.concat([perf_plot_AB2, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 3:
                            perf_plot_AB3 = pd.concat([perf_plot_AB3, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                    elif row['Protocol'] == 'AB-CD':
                        if row['ProtocolDay'] == 1:   
                            perf_plot_CD1 = pd.concat([perf_plot_CD1, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 2:
                            perf_plot_CD2 = pd.concat([perf_plot_CD2, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
                        elif row['ProtocolDay'] == 3:
                            perf_plot_CD3 = pd.concat([perf_plot_CD3, pd.DataFrame([{'Animal': row['Animal'], 'Gender': row['Gender'], 'Genotype': row['Genotype'], 'Block': bb, 'RewardRate': row['RewardRate'][bb], 'd': row['d'][bb]}])], ignore_index=True)
            
        for sex in Sexes:
            plot_learning_curve(perf_plot_AB1[perf_plot_AB1['Gender'] == sex], save_name = 'AB1_rewardrate_' + sex, 
                                value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'AB1 Reward Rate '+ sex + ' ' + self.strain)
            plot_learning_curve(perf_plot_AB2[perf_plot_AB2['Gender'] == sex], save_name = 'AB2_rewardrate_' + sex, 
                        value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'AB2 Reward Rate '+ sex + ' ' + self.strain)
            plot_learning_curve(perf_plot_AB3[perf_plot_AB3['Gender'] == sex], save_name = 'AB3_rewardrate_' + sex,
                        value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'AB3 Reward Rate '+ sex + ' ' + self.strain)
            plot_learning_curve(perf_plot_CD1[perf_plot_CD1['Gender'] == sex], save_name = 'CD1_rewardrate_' + sex,
                        value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'CD1 Reward Rate '+ sex + ' ' + self.strain)
            plot_learning_curve(perf_plot_CD2[perf_plot_CD2['Gender'] == sex], save_name = 'CD2_rewardrate_' + sex,
                        value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'CD2 Reward Rate '+ sex + ' ' + self.strain)
            plot_learning_curve(perf_plot_CD3[perf_plot_CD3['Gender'] == sex], save_name = 'CD3_rewardrate_' + sex,
                        value_col = 'RewardRate', trial_col = 'Block', summary_path = self.summary,
                                title = 'CD3 Reward Rate '+ sex + ' ' + self.strain)

        # plot_learning_curve(perf_plot_AB1, save_name = 'AB1_dprime', 
        #     value_col = 'd', trial_col = 'Block')
        # plot_learning_curve(perf_plot_AB2, save_name = 'AB2_dprime', 
        #     value_col =   'd', trial_col = 'Block')
        plt.close('all')

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
                startTime = -4.99
                endTime = 4.99
                aligned_t = np.arange(startTime, endTime,0.01)

                # video timestamp
                videoTS = pd.read_csv(self.data_index['behTimeStamp_aligned'][ii], header=0)
                header = ['TimeStamp', 'AlignedTimeStamp']
                videoTS.columns = header
                nFrames = videoTS.shape[0]
                FrameIdx = np.arange(nFrames)
                align_events = ['center_in', 'side_in']
                for event in align_events:
                    aligned_keypoints[event] = {}
                    aligned_keypoints[event]['frameID'] = np.full((len(aligned_t), nTrials), np.nan)
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


                                # save once (e.g. for nose only, since frame IDs are shared)
                                if bp == 'nose':
                                    sig_t = (videoTS['AlignedTimeStamp'][timeMask] - t_middle).to_numpy()
                                    frameIDTmp = FrameIdx[timeMask]
                                    nearest_idx = np.abs(sig_t[:, None] - aligned_t[None, :]).argmin(axis=0)
                                    nearest_frameID = frameIDTmp[nearest_idx]
                                    aligned_keypoints[event]['frameID'][:, tt] = nearest_frameID

                choice = np.array(behDF['actions'])
                prev_choice = np.concatenate([[np.nan], choice[0:-1]])

                # load one frome of the video for background
                videoPath = self.data_index['behRecording'][ii]
                frame = iio.imread(videoPath, index=0)

                # look for trial length, plot the trajectory (focus on short trials)

                # from last trial side_out to next trial side_in

                plt.figure()
                plt.imshow(frame)
                trialNum = 5
                x_trial = aligned_keypoints['center_in']['head']['x'][:, trialNum]
                y_trial = aligned_keypoints['center_in']['head']['y'][:, trialNum]
                valid = np.isfinite(x_trial) & np.isfinite(y_trial) & np.isfinite(aligned_t)
                points = np.column_stack((x_trial[valid], y_trial[valid]))
                if points.shape[0] > 1:
                    segments = np.stack((points[:-1], points[1:]), axis=1)
                    lc = LineCollection(segments, cmap='viridis', linewidth=5)
                    lc.set_array(aligned_t[valid][:-1])
                    plt.gca().add_collection(lc)
                
                # plot a given frame
                plt.figure()
                timePoint = 800
                trialNum = 0
                
                
                plt.scatter(aligned_keypoints['center_in']['left foot']['x'][timePoint,trialNum], 
                            aligned_keypoints['center_in']['left foot']['y'][timePoint,trialNum], color='red')
                plt.scatter(aligned_keypoints['center_in']['right foot']['x'][timePoint,trialNum], 
                            aligned_keypoints['center_in']['right foot']['y'][timePoint,trialNum], color='blue')
                # load the frame
                # fine the frameIdx
                frameIdx = aligned_keypoints['center_in']['frameID'][timePoint,trialNum]
                frame = iio.imread(videoPath, index=frameIdx)
                plt.imshow(frame)   

                import numpy as np
                import matplotlib.pyplot as plt
                from scipy.stats import gaussian_kde
                from scipy.ndimage import gaussian_filter1d
                from matplotlib.animation import FuncAnimation
                from matplotlib.colors import Normalize

                # X,Y: nTrials × nTime
                # t: time vector

                # ---- density from all trials ----
                X = aligned_keypoints['center_in']['head']['x']
                Y = aligned_keypoints['center_in']['head']['y']
                X = gaussian_filter1d(X, sigma=1, axis=1)
                Y = gaussian_filter1d(Y, sigma=1, axis=1)
                t = aligned_t
                # ---------------------------------
                # fixed plot limits
                # ---------------------------------

                xmin = np.nanmin(X)
                xmax = np.nanmax(X)

                ymin = np.nanmin(Y)
                ymax = np.nanmax(Y)

                xx, yy = np.mgrid[
                    xmin:xmax:100j,
                    ymin:ymax:100j
                ]

                grid = np.vstack([xx.ravel(), yy.ravel()])

                # ---------------------------------
                # setup figure
                # ---------------------------------

                fig, ax = plt.subplots(figsize=(8,8))
                ax.imshow(frame)
                density_im = ax.imshow(
                    np.zeros(xx.shape),
                    extent=[xmin,xmax,ymin,ymax],
                    origin='lower',
                    aspect='auto',
                    animated=True
                )

                trajectory_line, = ax.plot(
                    [],
                    [],
                    lw=3,
                    color='white'
                )

                time_text = ax.text(
                    0.05,
                    0.95,
                    '',
                    transform=ax.transAxes,
                    color='white',
                    fontsize=14
                )

                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')

                # average trajectory
                x_mean = np.nanmean(X,axis=1)
                y_mean = np.nanmean(Y,axis=1)

                # ---------------------------------
                # animation function
                # ---------------------------------

                window = 5   # temporal smoothing window

                def update(frame):

                    start = max(0, frame-window)
                    end = min(len(t), frame+window)

                    x_now = X[start:end,:].flatten()
                    y_now = Y[start:end,:].flatten()

                    valid = (
                        ~np.isnan(x_now)
                        &
                        ~np.isnan(y_now)
                    )

                    x_now = x_now[valid]
                    y_now = y_now[valid]

                    if len(x_now) > 10:

                        kde = gaussian_kde(
                            np.vstack([x_now,y_now])
                        )

                        density = kde(grid).reshape(xx.shape)

                        density_im.set_array(
                            density.T
                        )

                        density_im.set_norm(
                            Normalize(
                                vmin=0,
                                vmax=np.percentile(
                                    density,
                                    99
                                )
                            )
                        )

                    trajectory_line.set_data(
                        x_mean[:frame],
                        y_mean[:frame]
                    )

                    time_text.set_text(
                        f'Time: {t[frame]:.2f} s'
                    )

                    return density_im,trajectory_line,time_text


                anim = FuncAnimation(
                    fig,
                    update,
                    frames=len(t),
                    interval=30,
                    blit=True
                )

                anim.save(
                    'trajectory_density.mp4',
                    writer='ffmpeg',
                    fps=30,
                    dpi=200
                )

                # left_choice_trials = choice==0
                # left_choice_prev = prev_choice==0
                # right_choice_prev = prev_choice==1
                # right_choice_trials = choice == 1
                # X= aligned_keypoints['center_in']['head']['x'][:,(left_choice_trials & left_choice_prev)]
                # Y= aligned_keypoints['center_in']['head']['y'][:,(left_choice_trials & left_choice_prev)]
                # # --- plot single trials ---
                # # for tt in range(nTrials):
                # #     plt.plot(X[:, tt], Y[:, tt],
                # #             color='gray', linewidth=1, alpha=0.5)

                # # --- compute average trajectory ---
                # x_mean = np.nanmean(X, axis=1)
                # y_mean = np.nanmean(Y, axis=1)
                # x_std = np.nanstd(X, axis=1)
                # y_std = np.nanstd(Y, axis=1)

                # # --- plot average ---
                # plt.plot(x_mean, y_mean,
                #         color='red', linewidth=3, label='mean')
                # for k in range(-1, 2):
                #     plt.plot(x_mean + k * x_std,
                #             y_mean + k * y_std,
                #             color='red',
                #             alpha=0.2)
                # # plot right-left trials
                # X= aligned_keypoints['center_in']['head']['x'][:,(left_choice_trials & right_choice_prev)]
                # Y= aligned_keypoints['center_in']['head']['y'][:,(left_choice_trials & right_choice_prev)]

                # x_mean = np.nanmean(X, axis=1)
                # y_mean = np.nanmean(Y, axis=1)

                # # --- plot average ---
                # plt.plot(x_mean, y_mean,
                #         color='blue', linewidth=3, label='mean')

                # plt.plot(np.nanmean(aligned_keypoints['center_in']['head']['x'][:,(right_choice_trials & left_choice_prev)], axis=1), 
                #          np.nanmean(aligned_keypoints['center_in']['head']['y'][:,(right_choice_trials & left_choice_prev)], axis=1),
                #         color='yellow', linewidth=3, label='mean')
                
                # plt.plot(np.nanmean(aligned_keypoints['center_in']['head']['x'][:,(right_choice_trials & right_choice_prev)], axis=1), 
                #          np.nanmean(aligned_keypoints['center_in']['head']['y'][:,(right_choice_trials & right_choice_prev)], axis=1),
                #         color='green', linewidth=3, label='mean')
                
                # plt.xlabel('X')
                # plt.ylabel('Y')
                # plt.title('Trajectory (single trials + mean)')
                # plt.axis('equal')
                # plt.legend()

                # plt.show()

                
                # plt.figure()
                # plt.imshow(frame)
                # sc = plt.scatter(aligned_keypoints['center_in']['head']['x'][:,0], aligned_keypoints['center_in']['head']['y'][:,0],
                #                   c=aligned_t, cmap='viridis', s=10)

                # plt.colorbar(sc, label='Time (s)')
                # plt.scatter(aligned_keypoints['center_in']['head']['x'][100,0], aligned_keypoints['center_in']['head']['y'][100,0], s=40)
                # x_smooth = moving_average(DLCdata['head']['x'], window=10)
                # y_smooth = moving_average(DLCdata['head']['y'], window=10)

                # # plot head position near the center_in time
                # nTrials = behDF.shape[0]
                # center_head_x = []
                # center_head_y = []

                # center_head_x_smoothed = []
                # center_head_y_smoothed = []
                # for tt in range(nTrials):
                #     center_in = behDF['center_in'][tt]
                #     center_out = behDF['center_out'][tt]
                #     timeMask = np.logical_and(videoTS['AlignedTimeStamp']<center_out, 
                #                               videoTS['AlignedTimeStamp']>center_in)
                #     center_head_x.append(np.array(DLCdata['head']['x'])[timeMask])
                #     center_head_y.append(np.array(DLCdata['head']['y'])[timeMask])
                #     center_head_x_smoothed.append(x_smooth[timeMask])
                #     center_head_y_smoothed.append(y_smooth[timeMask])


                # center_x = np.concatenate(center_head_x)
                # center_y = np.concatenate(center_head_y)
                # center_x_smoothed = np.concatenate(center_head_x_smoothed)
                # center_y_smoothed = np.concatenate(center_head_y_smoothed)


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

    def __init__(self, root_file, strain):
        super().__init__(root_file, strain)
        self.make_dataIndex()

        self.behavior = 'Rotarod'

        strain_parts = strain.split('_')
        if any(part in ['Cntnap2'] for part in strain_parts):
            self.Mut = 'KO'
        elif any(part in ['TSC2', 'Shank3B', 'Chd8', 'Syngap', 'Scn2A'] for part in strain_parts):
            self.Mut = 'HET'
        elif any(part in ['Nlgn3'] for part in strain_parts):
            self.Mut = 'HEM'
        
    def make_dataIndex(self):
        # Create a data index, each row is a session
        # build the data index from rr_result
        rr_results_path = os.path.join(self.data, 'RR_results.csv')
        rr_results = pd.read_csv(rr_results_path)

        if rr_results.shape[0] > 0:
            self.data_index = pd.DataFrame()
            self.data_index['Animal'] = rr_results['AnimalID']
            self.data_index['Trial'] = rr_results['Trial']
            self.data_index['Date'] = rr_results['Date']
            self.data_index['Performance'] = rr_results['Performance']
            self.data_index['FallByTurning'] = rr_results['FBT']
            self.data_index['Genotype'] = rr_results['Genotype']
            self.data_index['Gender'] = rr_results['Gender']



            # if 'HET' in self.data_index['Genotype'].values:
            #     self.Mut = 'HET'
            # elif 'KO' in self.data_index['Genotype'].values:
            #     self.Mut = 'KO'
            self.WT = 'WT' 

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
                date = self.data_index['Date'][aidx]
                dataFolder = os.path.join(self.data, aa, 'Rotarod', 'BehavioralRecording',aa+'_'+str(date)[2:])
                trialNum = self.data_index['Trial'][aidx]
                if os.path.exists(dataFolder):
                    filePatternSpeed = aa + '*speed*.csv'
                    filePatternDLC = aa + '*DLC_resnet*.csv'
                    filePatternVideo = aa + '*.avi'
                    filePatternTimestamp = aa + '*timeStamp*.csv'

                    speedCSV = glob.glob(os.path.join(dataFolder, filePatternSpeed))
                    timeStampCSV = glob.glob(os.path.join(dataFolder, filePatternTimestamp))
                    videoFiles = glob.glob(os.path.join(dataFolder, filePatternVideo))
                    DLCFiles = glob.glob(os.path.join(dataFolder, filePatternDLC))
                    num_files = len(videoFiles)

                    if num_files>0:
                        # match the sessions: ASDxxx followed by optional middle part, 
                        # then trial(trialNum), optional underscore, and date YYYY-MM-DDTHH...
                        DLC_ID = [ID for ID in range(len(DLCFiles)) if aa in DLCFiles[ID] and 'trial'+str(trialNum) in DLCFiles[ID]]
                        if len(DLC_ID)>0:
                            DLC_results[aidx] = DLCFiles[DLC_ID[0]]
                        else:
                            DLC_results[aidx] = None
                        speed_ID = [ID for ID in range(len(speedCSV)) if aa in speedCSV[ID] and 'trial'+str(trialNum) in speedCSV[ID]]

                        if len(speed_ID)>0:
                            Rod_speed[aidx] = speedCSV[speed_ID[0]]
                        else:
                            Rod_speed[aidx] = None

                        timeStamp_ID = [ID for ID in range(len(timeStampCSV)) if aa in timeStampCSV[ID] and 'trial'+str(trialNum) in timeStampCSV[ID]]
                        if len(timeStamp_ID)>0:
                            timeStamp[aidx] = timeStampCSV[timeStamp_ID[0]]
                        else:
                            timeStamp[aidx] = None

                        video_ID = [ID for ID in range(len(videoFiles)) if aa in videoFiles[ID] and 'trial'+str(trialNum) in videoFiles[ID]]
                        if len(video_ID)>0:
                            video[aidx] = videoFiles[video_ID[0]]                    
                        else:      
                            video[aidx] = None

                        #stage.append(matches[0])
                        analysis[aidx] = os.path.join(self.analysis, aa,'Rotarod', 'Behavior', aa+'_'+str(date)[2:], 'trial'+str(trialNum))



            self.data_index['DLC'] = DLC_results
            self.data_index['Video'] = video
            self.data_index['Rod_speed'] = Rod_speed
            self.data_index['AnalysisPath'] = analysis
            self.data_index['BehTimestamp'] = timeStamp


            self.nSubjects = len(self.Animals)
            #sorted_df = self.dataIndex.sort_values(by=['Animal', 'Trial'])
            #sorted_df = sorted_df.reset_index(drop=True)
            #self.data=sorted_df
            self.nSessions = len(self.data_index['Animal'])

    def plot_performance(self):
        # looks for gender groups
        sexes = self.data_index['Gender'].unique()

          
        for sex in sexes:
            perf_df = self.data_index[['Animal', 'Genotype', 'Gender','Trial', 'Performance', 'FallByTurning']].copy()
            #if 'Cntnap' in self.strain:
            perf_df = perf_df[np.logical_or(perf_df['Genotype'] == self.Mut, perf_df['Genotype'] == self.WT)]
            # only consider KO for now
            perf_df = perf_df[perf_df['Gender'] == sex]
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
                'Gender': 'sex',
                'Performance': 'performance'
            })
            perf_df['trial'] = pd.Categorical(perf_df['trial'], categories=np.sort(perf_df['trial'].dropna().unique()), ordered=True)
            genotype_order = [g for g in ['WT', 'HET', 'KO'] if g in set(perf_df['genotype'].dropna())]
            genotype_order += [g for g in perf_df['genotype'].dropna().unique() if g not in genotype_order]
            perf_df['genotype'] = pd.Categorical(perf_df['genotype'], categories=genotype_order)
            genotype_counts = perf_df.dropna(subset=['subject', 'genotype']).groupby('genotype', observed=True)['subject'].nunique()

            os.makedirs(self.summary, exist_ok=True)

            figure_label = 'Rotarod ' + self.strain + ' ' + sex
            perf_df.to_csv(os.path.join(self.summary, f'{figure_label}_performance.csv'), index=False)

            #stats_df,_,_ = run_learning_FDA(perf_df, summary_path = self.summary,save_name = f'{figure_label}_FDA_stats.csv' )

            # ------------------------------------------------------------
            # Robustly mark failed trials (FallByTurning)
            # ------------------------------------------------------------
            exclude_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
            exclude_text = perf_df['FallByTurning'].astype(str).str.lower().isin(
                ['true', '1', 'yes']
            )

            perf_df.loc[exclude_numeric | exclude_text, 'performance'] = np.nan

            # if performance larger than 300, make it 300
            perf_df.loc[perf_df['performance'] > 300, 'performance'] = 300
            plot_learning_curve(perf_df, save_name = figure_label, 
                        value_col = 'performance', trial_col = 'trial', summary_path = self.summary,
                                title = figure_label)

            #return perf_df, stats_df
    
    def load_DLC_data(self):
        # Load rotarod behavior data from file
        DLC_obj= []

        for s in range(self.nSessions):
            analysisPath = self.data_index['AnalysisPath'][s]

            filePath = self.data_index['DLC'][s]
            videoPath = self.data_index['Video'][s]
            rodPath = self.data_index['Rod_speed'][s]
            fps = self.data_index['BehTimestamp'][s]
            #fps = 50
            if fps is None or len(fps)==0:
               dlc = None
            elif filePath is None:
                dlc = None
            else:
                dlc = DLCSession(filePath, videoPath, rodPath, analysisPath, fps)  
            DLC_obj.append(dlc)

        self.data_index['DLC_obj'] = DLC_obj
        #self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSessions)
        self.WTIdx = animalIdx[self.data_index['Genotype'] == self.WT]
        self.MutIdx = animalIdx[self.data_index['Genotype'] == self.Mut]


        self.startVoltage = [4.45, 40] # 5 rpm = 0.273 V
        self.endVoltage = [8.90, 80]
        self.rod_a = (self.endVoltage[1] - self.startVoltage[1]) / (self.endVoltage[0] - self.startVoltage[0])
        self.rod_b = self.endVoltage[1] - self.endVoltage[0] * self.rod_a
 
    def align_timeStamps(self):
        # preprocess data, align the video with rod speed.
        # we will need to manually label the number of frame when rod start to turn
        # smooth the DLC labelling and save the corrected DLC

        #stampCSV = os.path.join(self.dataFolder,'Videos', 'timeStamp.csv')
        #timeStamp = pd.read_csv(stampCSV)
        #self.data['rodData'] = [[] for x in range(self.nSessions)]

        # check if DLC data exists

        savefigpath = os.path.join(self.analysis, 'StartPoint_rodSpeed')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)

        for ss in range(self.nSessions):
            savefigname = os.path.join(savefigpath, 'Start point for ' + str(self.data_index['Animal'][ss]) + ' trial' + str(self.data_index['Trial'][ss])+'.png')
            # check if fig exist
            DLC_obj = self.data_index['DLC_obj'][ss]
            if DLC_obj is not None:
                savedataname = os.path.join(DLC_obj.analysis, 'smoothed_rodSpeed.csv')
                if not os.path.exists(savedataname):

                    DLC_obj = self.data_index['DLC_obj'][ss]

                    # Generate a sample signal with change points
                    if DLC_obj.data['rodSpeed'] is not None:
                        # Create a change point detection object using a specific algorithm (e.g., 'Pelt' or 'Binseg')
                        signal = DLC_obj.data['rodSpeed']/100

                        # downsample it

                        algo = rpt.Pelt(model="l2").fit(signal)
                        # Predict the change points
                        predicted_bkps = algo.predict(pen=3)

                        # Display results
                        # change the signal to rod speed

                        rodSpeed = signal*self.rod_a+self.rod_b

                        max_jump = (80-5)/(60*5*4)
                        # Smooth the signal
                        smoothed_signal = np.copy(rodSpeed)
                        #running average first
                        tempSpeed = rodSpeed
                        #plt.figure()
                        #plt.plot(rodSpeed)
                        if rodSpeed[0] == 0 and rodSpeed[-1] == 0:
                            # steady state has been recorded
                            startRange = predicted_bkps[0]
                            endRange = predicted_bkps[-2]
                        elif rodSpeed[0] == 0 and rodSpeed[-1] > 0:
                            startRange = predicted_bkps[0]
                            endRange = len(rodSpeed)
                        elif  rodSpeed[0] > 0 and rodSpeed[-1] == 0:
                            # steady state hasn't been recorded
                            startRange = 0
                            endRange = predicted_bkps[-2]
                        else:
                            startRange = 0
                            endRange = len(rodSpeed)

                        for sss in range(0,60,10):
                            windowSize = 4+sss*1

                            for i in range(startRange, endRange):
                                if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
                                elif i <= startRange + windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
                                elif i >= endRange - windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])
                            tempSpeed = smoothed_signal
                            #plt.plot(tempSpeed)
                            # jump = smoothed_signal[i] - smoothed_signal[i - 1]
                            # if abs(jump) > max_jump:
                            #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
                                #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2

                        algo = rpt.Pelt(model="l2").fit(smoothed_signal)
                        # Predict the change points
                        new_predicted_bkps = algo.predict(pen=1)

                        # smooth one more round with the new predicted change point
                        #plt.figure()
                        #plt.plot(smoothed_signal)
                        for sss in range(0,60,10):
                            windowSize = 4+sss*1
                            for i in range(startRange, endRange):
                                if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
                                elif i <= startRange + windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
                                elif i >= endRange - windowSize/2:
                                    smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])

                            # jump = smoothed_signal[i] - smoothed_signal[i - 1]
                            # if abs(jump) > max_jump:
                            #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
                                #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2
                        rodTime = DLC_obj.data['rodT']

                        # algo = rpt.Pelt(model="l2").fit(smoothed_signal)
                        # # Predict the change points
                        # new_predicted_bkps = algo.predict(pen=170)

                        # find the point when rod speed start to increase with first derivative
                        dx = np.diff(smoothed_signal)/np.diff(rodTime)
                        k=200
                        runIdx = np.where(np.convolve((dx > 0.1).astype(int), np.ones(k, dtype=int), 'valid') == k)[0][0]

                        startIdx = np.where(smoothed_signal>0.5)[0][0]
                        endIdx = np.where(smoothed_signal>0.5)[0][-1]
                        # save the running speed and voltage
                        # save the smoothed_signal somewhere
                        saveData={}
                        saveData['raw'] = signal
                        saveData['smoothed'] = smoothed_signal
                        saveData['time'] = rodTime
                        if rodSpeed[0] ==0:
                            saveData['Start'] = np.zeros((len(rodTime)))+rodTime[startIdx]
                            saveData['Run'] = np.zeros((len(rodTime)))+rodTime[runIdx]
                        else:
                            saveData['Start'] = np.full((len(rodTime)),np.nan)
                            saveData['Run'] = np.zeros((len(rodTime)))+rodTime[runIdx]
                        savedf= pd.DataFrame(saveData)
                        savedf.to_csv(savedataname)

                        plt.figure()
                        plt.plot(rodTime,rodSpeed)
                        plt.plot(rodTime,smoothed_signal)
                        plt.scatter(rodTime[startIdx], 0, s=200)
                        plt.scatter(rodTime[runIdx], 0, s=200)
                        plt.scatter(rodTime[endIdx],0, s=200)
                        plt.title('Start point for ' + str(self.data_index['Animal'][ss]) + ' trial' + str(self.data_index['Trial'][ss]))
                        #plt.show()
                        plt.savefig(savefigname)
                        plt.close()
                    # else:
                    #     saveData = pd.read_csv(savedataname)


                
                        # if t0 and saveData['Run'][0] is close enough (1s), use t0
                        #if t0 and abs(t0 - saveData['Run'][0]) < 1:
                        #saveData['Run'] = np.zeros((len(saveData['time']))) + t0

                        self.data_index['DLC_obj'][ss].data['rodSpeed_smoothed'] = saveData['smoothed']
                        self.data_index['DLC_obj'][ss].data['rodStart'] = saveData['Start']
                        self.data_index['DLC_obj'][ss].data['rodRun'] = saveData['Run']
                    # load it somewhere?

                        #%% find the point when animal turn around

                        # clean the data up:
                        # 1. identify jumps in the DLC data, then infer the correct estimation based on previous and next frame
                        # 2. add a mask on over turning frames
                        # if DLC file exists:

                        #%% for every keypoint find the jumps
                        # orig_DLC = self.data_index['DLC'][ss]
                        # corr_DLC = os.path.join(os.path.dirname(orig_DLC), os.path.basename(orig_DLC)[:-4] + '_corrected.csv')
                        # self.data_index.loc[ss, 'Corrected_DLC'] = corr_DLC

                        # if not os.path.exists(corr_DLC):
                        #     corrected_data = {}
                        #     for bp in kp_list:
                        #         x_data = np.array(tempData[bp]['x'])
                        #         y_data = np.array(tempData[bp]['y'])
                        #         p_data = np.array(tempData[bp]['p'])

                        #     # Calculate differences between consecutive frames
                        #         dx = np.abs(np.diff(x_data))
                        #         dy = np.abs(np.diff(y_data))
                        #         distance = np.sqrt(dx**2 + dy**2)
                        #         # Define a threshold for detecting jumps (this may need to be adjusted)
                        #         jump_threshold = 200  # pixels

                        #         # find jump points (where distance deviate largely from its neighbors)
                        #         k = 5
                        #         jump_frames = []

                        #         for i in range(len(x_data)):
                        #             left = max(0, i-k)
                        #             right = min(len(x_data), i+k+1)
                        #             neigh_x = np.delete(x_data[left:right], i-left)
                        #             neigh_y = np.delete(y_data[left:right], i-left)

                        #             med_x = np.median(neigh_x)
                        #             med_y = np.median(neigh_y)

                        #             dist = np.sqrt((x_data[i] - med_x)**2 + (y_data[i] - med_y)**2)
                        #             if dist > jump_threshold:
                        #                 jump_frames.append(i)  # +1 to correct index after diff

                        #         # Correct jumps by interpolation with forward and backward 5 frames that did not jump
                        #         interp_window = 5
                        #         for jf in jump_frames:
                        #             before = []
                        #             after = []
                        #             before_i = jf-1
                        #             while before_i >= 0 and len(before) < interp_window:
                        #                 if before_i not in jump_frames:
                        #                     before.append(before_i)
                        #                 before_i -= 1
                        #             after_i = jf+1
                        #             while after_i < len(x_data) and len(after) < interp_window:
                        #                 if after_i not in jump_frames:
                        #                     after.append(after_i)
                        #                 after_i += 1
                        #             valid_idx = before[::-1] + after 
                        #             xi = np.array(valid_idx)
                        #             yi_x = x_data[xi]
                        #             yi_y = y_data[xi]
                        #             yi_p = p_data[xi]
                        #             # interpolate at frame jf
                        #             x_data[jf] = np.interp(jf, xi, yi_x)
                        #             y_data[jf] = np.interp(jf, xi, yi_y)
                        #             p_data[jf] = np.interp(jf, xi, yi_p)

                        #         # set a new matrix to save interpolated data
                        #         corrected_data[bp] = {'x': x_data, 'y': y_data, 'p': p_data}

                        #         # save the corrected_data to a new csv file

                        #         # load the original csv to get the header
                        #         orig_df = pd.read_csv(orig_DLC, header=[0,1,2], index_col=0)
                        #         corrected_df = pd.DataFrame(index=orig_df.index, columns=orig_df.columns)
                        #         scorer = orig_df.columns.levels[0][0]
                        #         for bp in kp_list:
                        #             corrected_df[(scorer, bp, 'x')] = corrected_data[bp]['x']
                        #             corrected_df[(scorer, bp, 'y')] = corrected_data[bp]['y']
                        #             corrected_df[(scorer, bp, 'likelihood')] = corrected_data[bp]['p']
                        #         corrected_df.to_csv(corr_DLC)

                else:
                    # load the data in savedataname
                    saveData = pd.read_csv(savedataname)
                    self.data_index['DLC_obj'][ss].data['rodSpeed_smoothed'] = np.array(saveData['smoothed'])
                    self.data_index['DLC_obj'][ss].data['rodStart'] = np.array(saveData['Start'])
                    self.data_index['DLC_obj'][ss].data['rodRun'] = np.array(saveData['Run'])   

                if self.data_index['DLC'][ss] is not None:
                    tempData = self.data_index['DLC_obj'][ss].data
                    ave_left_rod_back = np.array([np.mean(np.array(tempData['rod_left_back']['x'])[np.array(tempData['rod_left_back']['p'])>0.95]),
                                    np.mean(np.array(tempData['rod_left_back']['y'])[np.array(tempData['rod_left_back']['p'])>0.95])])
                    ave_right_rod_back = np.array([np.mean(np.array(tempData['rod_right_back']['x'])[np.array(tempData['rod_right_back']['p'])>0.95]),
                                    np.mean(np.array(tempData['rod_right_back']['y'])[np.array(tempData['rod_right_back']['p'])>0.95])])
                    ave_center_rod_back = (ave_left_rod_back+ave_right_rod_back)/2
                    self.data_index['DLC_obj'][ss].data['left_rod_back'] = ave_left_rod_back
                    self.data_index['DLC_obj'][ss].data['right_rod_back'] = ave_right_rod_back
                    self.data_index['DLC_obj'][ss].data['center_rod_back'] = ave_center_rod_back

                    ave_left_rod_front = np.array([np.mean(np.array(tempData['rod_left_front']['x'])[np.array(tempData['rod_left_front']['p'])>0.95]),
                                    np.mean(np.array(tempData['rod_left_front']['y'])[np.array(tempData['rod_left_front']['p'])>0.95])])
                    ave_right_rod_front = np.array([np.mean(np.array(tempData['rod_right_front']['x'])[np.array(tempData['rod_right_front']['p'])>0.95]),
                                    np.mean(np.array(tempData['rod_right_front']['y'])[np.array(tempData['rod_right_front']['p'])>0.95])])
                    ave_center_rod_front = (ave_left_rod_front+ave_right_rod_front)/2

                    self.data_index['DLC_obj'][ss].data['left_rod_front'] = ave_left_rod_front
                    self.data_index['DLC_obj'][ss].data['right_rod_front'] = ave_right_rod_front
                    self.data_index['DLC_obj'][ss].data['center_rod_front'] = ave_center_rod_front

                    # estimations on the left of 'right_rod_back' is in back area
                    # on the right of 'right rod front' is in front area
                    # plot a 1/0 mask to show each keypoints where they belong
                    kp_list = tempData['bodyparts']
                    viewMask = np.zeros((len(kp_list), len(tempData['rod_right_back']['x'])))
                    # 1 for back 0 for front
                    for idx,kp in enumerate(kp_list):
                    # on the left of
                        viewMask[idx,:] = np.array(tempData[kp]['x']) < ave_right_rod_back[0]

                    # plot frame with keypoints
                    # frame_num = 6180
                    # curr_frame = read_video(self.data['DLC_obj'][ss].videoPath, frame_num, ifgray=False)
                    # plt.figure()
                    # plt.imshow(curr_frame)
                    # cmap = cm.get_cmap('viridis', len(kp_list))
                    # for kp in kp_list:
                    #     plt.scatter(tempData[kp]['x'][frame_num], tempData[kp]['y'][frame_num], c=cmap(kp_list.index(kp)), s=200,label = kp)
                    #
                    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    # plot number of back/front keypoints that is actually in back/front view
                    back_kp = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
                    front_kp = ['spine 1', 'left ear', 'right ear', 'nose', 'left hand', 'right hand']
                    viewNumber = np.zeros((3, len(tempData['rod_right_back']['x'])))
                    for kp in kp_list:
                        if kp in back_kp:
                            viewNumber[0,:] = viewNumber[0,:] + viewMask[kp_list.index(kp),:]
                        elif kp in front_kp:
                            viewNumber[1,:] = viewNumber[1,:] + 1-viewMask[kp_list.index(kp),:]
                    viewNumber[2,:] = (viewNumber[0,:] + viewNumber[1,:])/(len(back_kp)+len(front_kp))

                    # plot number of back/front keypoints that is actually in back/front view
                    # need to set some threshold to identify when the animal turn around
                    # consistently smaller 50% for longer than 3 seconds?
                    p_thresh = 0.6
                    min_duration = 1 # seconds
                    below_threshold = viewNumber[2,:] < p_thresh

                    # Initialize start and end indices list
                    segments = []
                    start_idx = None

                    for i in range(len(viewNumber[2,:])):
                        if below_threshold[i]:
                            # Start of a new segment
                            if start_idx is None:
                                start_idx = i
                        else:
                            # End of a segment
                            if start_idx is not None:
                                # Calculate segment duration
                                duration = tempData['time'][i - 1] - tempData['time'][start_idx]
                                if duration >= min_duration:
                                    segments.append((start_idx, i - 1))
                                start_idx = None

                    # Check if the last segment meets the minimum duration
                    if start_idx is not None:
                        duration = tempData['time'][-1] - tempData['time'][start_idx]
                        if duration >= min_duration:
                            segments.append((start_idx, len(viewNumber[2,:]) - 1))

                    saveFolder = os.path.join(self.analysis, 'TurnAround')
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                    savepath = os.path.join(saveFolder,
                                'turning period_' + self.data_index['Animal'][ss] + '_Trial ' + str(self.data_index['Trial'][ss]) + '.png')

                    if not os.path.exists(savepath):

                        fig, ax = plt.subplots(3,1, figsize=(16, 8),
                                            gridspec_kw={'height_ratios': [3, 1, 1]})  # Adjust figure size for visibility
                        ax[0].imshow(viewMask, cmap='gray', aspect='auto', interpolation='none')
                        ax[0].set_yticks(ticks=np.arange(len(kp_list)), labels=kp_list)
                        #x_tick_interval = 50
                        #x_positions = np.where((tempData['time'] % x_tick_interval) < (x_tick_interval / len(tempData['time'])))[0]
                        #x_labels = [f"{int(tempData['time'][idx])}" for idx in x_positions]
                        #ax[0].set_xticks(ticks=x_positions, label                   ax[1].plot(tempData['time'], viewNumber[0,:],linewidth=1)
                        ax[1].plot(tempData['time'], viewNumber[1,:],linewidth=1)
                        ax[1].plot(tempData['time'], viewNumber[2,:],linewidth=1)
                        for tt in segments:
                            ax[1].axvspan(tempData['time'][tt[0]], tempData['time'][tt[1]], color='red', alpha=1)

                        ax[2].plot(tempData['rodT'],saveData['smoothed'])
                        # save the figure
                        fig.savefig(savepath, dpi=300, bbox_inches='tight')
                        plt.close()

                    self.data_index['DLC_obj'][ss].data['turning_period'] = segments
                    self.data_index['DLC_obj'][ss].data['turning_mask'] = np.full((len(self.data_index['DLC_obj'][ss].data['time'])), 1)
                    for seg in segments:
                        self.data_index['DLC_obj'][ss].data['turning_mask'][seg[0]:seg[1]+1] = 0


    def align_with_calcium(self, calcium_timestamps):
        # Align rotarod behavior timestamps with calcium imaging timestamps
        pass

    def stride_session(self, front_kp, back_kp):
        #todo: go over each trial and run get_stride for them
        for idx, obj in enumerate(self.data_index['DLC_obj']):
            # check if DLC exist

            if (self.data_index['DLC'][idx] is not None) and (len(self.data_index['DLC'][idx])>0):
                obj.get_stride(front_kp, back_kp, self.data_index.iloc[idx])

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
        genotype = self.Genotypes

        for idx, obj in enumerate(self.data_index['DLC_obj']):
            animal = self.data_index['Animal'][idx]
            trialIdx = self.data_index['Trial'][idx]-1
            animalIdx = self.Animals.index(animal)

            if (self.data_index['DLC'][idx] is not None) and (obj is not None):
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
                
                if not self.data_index['FallByTurning'][np.logical_and(self.data_index['Animal']==animal,
                                                 self.data_index['Trial']==trialIdx+1)].any():
                    amp_std['perf'][animalIdx, trialIdx] = self.data_index['Performance'][np.logical_and(self.data_index['Animal']==animal,
                                                                                            self.data_index['Trial']==trialIdx+1)].values[0]
                
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
                        truncatedCorr.loc[nanMask, key] = np.nan

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
            ko_mask = genotype == 'HET'

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
            savefigpath = os.path.join(self.summary, 'Performance vs ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.summary, 'Performance vs ' + key + ' Amplitude SD.svg')
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
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'HET'])

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

            p_genotype = get_p('genotype[T.HET]')
            p_genotype_speed = get_p('genotype[T.HET]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.HET]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'HET', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

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

            savefigpath = os.path.join(self.summary, 'Changes of ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.summary, 'Changes of  ' + key + ' Amplitude SD.svg')
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
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'HET'])

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

            p_genotype = get_p('genotype[T.HET]')
            p_genotype_speed = get_p('genotype[T.HET]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.HET]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'HET', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

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

            savefigpath = os.path.join(self.summary, 'Changes of ' + key + ' Average Amplitude.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.summary, 'Changes of  ' + key + 'Average Amplitude.svg')
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
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'HET'])

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

            p_genotype = get_p('genotype[T.HET]')
            p_genotype_speed = get_p('genotype[T.HET]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.HET]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'HET', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

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

            savefigpath = os.path.join(self.summary, 'Changes of ' + key + ' Average Frequency.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.summary, 'Changes of  ' + key + 'Average Frequency.svg')
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

            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'HET']
            colors = {'WT': 'black', 'HET': 'red'}

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

            savefigpath = os.path.join(self.summary, 'Changes of ' + key + '.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.summary, 'Changes of  ' + key + '.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average amplitude/frequency at 5-20 RPM within trial 1-3, 4-6, 7-9, and 10-12
        
    def process_for_moseq(self):
        # stride analysis for rotarod behavior
        # in situations where individual syllables jumped to the opposite side
        # infer the coordinate with previous and post frames
        # also remove the part when animal turns around

        savefilefolder = os.path.join(self.root_path,'DLCforMoseq')
        saveorigfolder = os.path.join(self.root_path, 'DLCforMoseq_origTS')
        if not os.path.exists(savefilefolder):
            os.makedirs(savefilefolder)
        if not os.path.exists(saveorigfolder):
            os.makedirs(saveorigfolder)

        for idx, obj in tqdm(enumerate(self.data_index['DLC_obj'])):
            animal = self.data_index['Animal'][idx]
            trialIdx = self.data_index['Trial'][idx]
            animalIdx = self.Animals.index(animal)

            if self.data_index['DLC_obj'][idx] is not None :
                # process video
                # remove the turning part, save the DLC and video 6in separate pieces
                #obj = self.data['DLC_obj'][idx]
                turning_mask = obj.data['turning_mask']
                turning_segments = obj.data['turning_period']
                # load the Stride_freq
                DLCCSV = self.data_index['DLC'][idx]
                df = pd.read_csv(DLCCSV)

                # remove the first the last 5 seconds for frames without animal on the rod
                timeStart = np.where(obj.data['time']>(obj.data['rodRun'][0]))[0][0] 
                timeOnRod = self.data_index['Performance'][np.logical_and(self.data_index['Animal']==animal,
                            self.data_index['Trial']==trialIdx)]
                timeEnd = np.where(obj.data['time']<(obj.data['rodRun'][0]+timeOnRod-8)[idx])[0][-1] 
                # isolate the time when animals turns around
                tempMask = np.logical_and(pd.to_numeric(df['scorer'][2:])>timeStart, pd.to_numeric(df['scorer'][2:])<=timeEnd)
                tempMask = [True, True]+ list(tempMask)

                file_path = os.path.normpath(DLCCSV)

                # Extract the base filename without extension
                filename = os.path.splitext(os.path.basename(file_path))[0]

                # go through turning segments to get a non-turning segments
                non_turning_segments = []
                if len(turning_segments) ==0:
                    non_turning_segments.append((0, len(obj.data['time'])-1))
                elif len(turning_segments) ==1:
                    non_turning_segments.append((0, turning_segments[0][0]-100))
                    non_turning_segments.append((turning_segments[0][1]+100, len(obj.data['time'])-1))
                else:
                    for turnIdx in range(len(turning_segments)):
                        if turnIdx==0:
                            start_frame = 0
                        else:
                            start_frame = turning_segments[turnIdx-1][1]+100
                        if turnIdx == len(turning_segments)-1:
                            end_frame = len(obj.data['time'])
                        else:
                            end_frame = turning_segments[turnIdx][0]-100  # remove about 1s before and after turning
                        non_turning_segments.append((start_frame, end_frame))

                for turnIdx in range(len(non_turning_segments)):
                    start_frame, end_frame = non_turning_segments[turnIdx]
                    nonTurningMask = np.zeros(len(obj.data['time']), dtype=bool)
                    nonTurningMask[start_frame:end_frame+1] = True
                    nonTurningMask = [True, True]+ list(nonTurningMask)
                    saveMask = np.logical_and(tempMask, nonTurningMask)

                    # check it is longer than 2 second
                    if sum(saveMask) > 100 and start_frame<end_frame:
                    # save the non-turning segment
                        df_segment = df[saveMask].copy()
                        orig_index = df_segment.iloc[:,0].copy()
                        # renumber it
                        col = df_segment.iloc[:, 0].astype(object)
                        col.iloc[2:] = pd.to_numeric(col.iloc[2:], errors='coerce')

                        df_segment[df_segment.columns[0]] = col

                        df_segment.iloc[2:,0] = np.arange(len(df_segment)-2)

                        # go over the bodyparts in the df_segment, if the bodypart jumped to far a way 
                        # to the other half of the frame, move it back 
                        correct_bodyparts(df_segment)
                        savefilepath = os.path.join(savefilefolder, filename+f"_clip{turnIdx}.csv")
                        df_segment.to_csv(savefilepath, index=False)
                        savetimeStamp = os.path.join(saveorigfolder, filename+f"origIndex_clip{turnIdx}.csv")
                        orig_index.to_csv(savetimeStamp, index=False)
                        # save the video segment
                        mask = saveMask.astype(int)

                        ones_idx = np.where(mask[2:] == 1)[0]

                        start_write = ones_idx[0]
                        end_write   = ones_idx[-1]
                        videoPath = self.data_index['Video'][idx]
                        #cap = cv2.VideoCapture(videoPath)
                        # container = av.open(videoPath)
                        # stream = container.streams.video[0]
                        #     # Determine FPS
                        # fps = float(stream.average_rate)  # frames per second
                        # width = stream.codec_context.width
                        # height = stream.codec_context.height

                        # Create output container and stream
                        savevideopath_segment = os.path.join(savefilefolder, filename +f'_clip{turnIdx}.mp4')
                        if not os.path.exists(savevideopath_segment):

                            cap = cv2.VideoCapture(videoPath)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            # Use MJPG or XVID for AVI; use mp4v or H264 for MP4
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
                            out = cv2.VideoWriter(savevideopath_segment, fourcc, fps, (width, height))

                            frame_idx = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                if frame_idx < start_write:
                                    frame_idx += 1
                                    continue
                                if frame_idx > end_write:
                                    break

                                out.write(frame)
                                frame_idx += 1

                            cap.release()
                            out.release()

                            # check if the saved video is corrupted (frames mismatch)
                            verify_cap = cv2.VideoCapture(savevideopath_segment)
                            verify_idx = 0
                            while True:
                                ret, frame = verify_cap.read()
                                if not ret:
                                    break
                                verify_idx += 1

                            verify_cap.release()
                            if verify_idx < (end_write - start_write + 1):
                                nonTurningMask = np.zeros(len(obj.data['time']), dtype=bool)
                                nonTurningMask[start_write:start_write+verify_idx] = True
                                nonTurningMask = [True, True]+ list(nonTurningMask)
                                saveMask = np.logical_and(tempMask, nonTurningMask)
                                df_segment = df[saveMask]
                                orig_index = df_segment.iloc[:,0].copy()
                                # renumber it
                                df_segment.iloc[2:,0] = np.arange(len(df_segment)-2)

                                savefilepath = os.path.join(savefilefolder, filename+f"_clip{turnIdx}.csv")
                                df_segment.to_csv(savefilepath, index=False)
                                savetimeStamp = os.path.join(saveorigfolder, filename+f"origIndex_clip{turnIdx}.csv")
                                orig_index.to_csv(savetimeStamp, index=False)
                                # if frames is corrupted, rewrite orig TS and DLC file to match the non-currupted video






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
