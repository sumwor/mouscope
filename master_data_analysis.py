# master script for analyzing behavioral and imaging data

from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()

run_odor = True
run_rotarod = True


if run_odor:
    #%% for odor behavioral data
    # analyze the TSC2 behavioral recording data
    strain = 'TSC2_adol'
    root_dir = os.path.join(r'Y:\HongliWang\Juvi_ASD Deterministic', strain)

    Odor = BehDataOdor(root_dir, strain)
    Odor.load_data()
    #Odor.session_analysis()
    #Odor.plot_performance()
    
    #%% model fitting (only implemented policy gradient for now)
    Odor.model_fitting(fit_mode='session')
    #Odor.model_fitting(fit_mode='concat')
    Odor.model_comparison()


    #Odor.plot_performance()
    #Odor.plot_response_times()

    Odor.align_timeStamps()
    Odor.DLC_analysis()

#%% for rotarod analysis

if run_rotarod:
 
    rotarod_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
    strains = ['TSC2', 'Shank3B', 'Nlgn3', 'Chd8', 'Cntnap2', 'Scn2A', 'Syngap']
    ages = ['adol', 'adult']

    #%% preprocess step need to run from time-to-time when we still actively collecting data
    # after data-collection, no need to run this step
    preprocess_rotarod = False
    if preprocess_rotarod:
        # preprocess step 1: read the rotarod performance data from google sheet
        # google sheet contains the most up-to-date data
        google_url = 'https://docs.google.com/spreadsheets/d/1LUcjvEakIwHhLN7hiLypfzwQK5Iod2zubOzcCtDorts/edit?gid=235330200#gid=235330200'
        #fetch_rotarod(google_url, rotarod_dir, strains, ages)
        
        # preprocess step 2: organize the rotarod videos and DLC files from raw video folders
        # to pipeline folders after clean up
        # move the videos to the corresponding folders in root_dir
        # find the corresponding dlc files from dlc_folder
        # move videos without DLC files to a separate folder for DLC labeling

        video_folder = r'Y:\HongliWang\Rotarod\rawRecordings_260622'
        #dlc_folder = r'Y:\HongliWang\Rotarod\Filtered_DLC'
        dlc_folder = r'Y:\HongliWang\Rotarod\DLC_training'
        # remove size 0 files and clean the filenames
        #clean_rotarod_videos(video_folder)

        organize_beh_videos(rotarod_dir, video_folder, dlc_folder)   


    
    #%% plot rotarod performance for each strain (separating age and gender)
    plot_perf = False
    if plot_perf:
        for strain in strains:
            for age in ages:
                root_dir = os.path.join(rotarod_dir, strain + '_' + age)
                strain_folder = strain + '_' + age

                # check if there is data
                Rotarod = BehDataRotarod(root_dir, strain_folder)
                #Rotarod.plot_performance()
                if len(Rotarod.Animals) > 0:
                    #Rotarod.load_DLC_data()
                    Rotarod.plot_performance()


    # check one strain first
    root_dir = r'Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol'
    strain_folder = 'TSC2_adol'

    # check if there is data
    Rotarod = BehDataRotarod(root_dir, strain_folder)
    
    # process the data and prepare them for Keypoint-moseq analysis
    Rotarod.load_DLC_data()
    Rotarod.align_timeStamps()
    
    Rotarod.process_for_moseq()



    back_keypoints = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
    front_keypoints = ['nose', 'left ear', 'right ear', 'left hand', 'right hand']

    Rotarod.stride_session(back_keypoints, front_keypoints)

    Rotarod.stride_summary()
