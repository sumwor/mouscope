# master script for analyzing behavioral and imaging data


from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()

run_odor = False
run_rotarod = True


if run_odor:
    #%% for odor behavioral data
    # analyze the TSC2 behavioral recording data
    strain = 'Scn2A_adol'
    root_dir = os.path.join(r'Y:\HongliWang\Juvi_ASD Deterministic', strain)

    Odor = BehDataOdor(root_dir, strain)
    Odor.load_data()
    #Odor.session_analysis()

    #%% model fitting (only implemented policy gradient for now)
    Odor.model_fitting(fit_mode='session')
    #Odor.model_fitting(fit_mode='concat')

    #Odor.plot_performance()
    #Odor.plot_response_times()

    Odor.align_timeStamps()
    Odor.DLC_analysis()

#%% for rotarod data
# for rotarod, need to run the code to update the RR_result from google sheet
# run fetch_rotrarod_log.py

if run_rotarod:
    google_url = 'https://docs.google.com/spreadsheets/d/1LUcjvEakIwHhLN7hiLypfzwQK5Iod2zubOzcCtDorts/edit?gid=235330200#gid=235330200'
    rotarod_data_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
    strains = ['TSC2', 'Shank3B', 'Nlgn3', 'Chd8', 'Cntnap2', 'Scn2A', 'Syngap']
    ages = ['adol', 'adult']

    #fetch_rotarod(google_url, rotarod_data_dir, strains, ages)
    # update resulf from the google sheet first
    
    for strain in strains:
        for age in ages:
            root_dir = os.path.join(rotarod_data_dir, strain + '_' + age)
            strain_folder = strain + '_' + age

            # check if there is data
            Rotarod = BehDataRotarod(root_dir, strain_folder)
            #Rotarod.plot_performance()
            if len(Rotarod.Animals) > 0:
                #Rotarod.load_DLC_data()
                Rotarod.plot_performance()

    Rotarod.align_timeStamps()

    back_keypoints = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
    front_keypoints = ['nose', 'left ear', 'right ear', 'left hand', 'right hand']

    Rotarod.stride_session(back_keypoints, front_keypoints)

    Rotarod.stride_summary()
