# code to process behavior files and align calcium data with behavior timestamps
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from datetime import datetime
import re

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
        self.Animals = self.AnimalInfo['AnimalID']
        self.Genotypes = self.AnimalInfo['Genotype']
        self.ImageCell = self.AnimalInfo['Cells']
        self.Hemisphere = self.AnimalInfo['hemisphere']


class BehDataOdor(BehData):

    def __init__(self, root_file):
        super().__init__(root_file)
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

                extra_columns = {
                    'ROIFile': '',         # store as JSON
                    'behRecording': '',                # list of .mp4 files
                    'behTimeStamp': '',                # list of timestamps
                    'AIMatrix': '',                    # list or array
                    'AITimsStamp': '',                 # list or array
                    'ImgTimeStamp': '',                # list or array
                    'ifCalImg': False,                 # boolean
                    'ifBehRecording': False,            # boolean
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
    
    def align_with_calcium(self, calcium_timestamps):
        # Align behavior timestamps with calcium imaging timestamps
        pass

    def performance(self):
        # call matlab function to plot the performance
        pass

    def model_fit(self):
        # call matlab function to fit the computational model
        pass

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

    Odor.session_behavior()


    #%% test code for rotarod behavior
    rotarod = BehDataRotarod(root_dir)

    rotarod.load_data()