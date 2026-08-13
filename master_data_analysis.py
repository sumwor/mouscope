# master script for analyzing behavioral and imaging data

# todo:
# 1. check if there is missing sessions by date
# 2. cross compare performance with digitized record

from sqlite3 import Time

from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()

run_odor = True
run_rotarod = True
count_animals = False

if run_odor:

    if count_animals:
        odor_dir = r'Y:\HongliWang\Odor'
        strains = ['TSC2', 'Shank3B', 'Nlgn3', 'Chd8', 'Cntnap2', 'Scn2A', 'Syngap']
        ages = ['adol', 'adult']

        #%% count the number of animals and trials for each strain, age, and genotype
        animal_count = {}
        for strain in strains:
            for age in ages:
                root_dir = os.path.join(odor_dir, strain + '_' + age)
                strain_folder = strain + '_' + age

                # read the animalLst.csv file
                animalList_file = os.path.join(root_dir,  'Data', 'AnimalList.csv')
                if os.path.exists(animalList_file):
                    animalList = pd.read_csv(animalList_file)


                    Genders = animalList['Gender'].unique()
                    session_length = animalList['Session_length'].unique()
                    Genotypes = animalList['Genotype'].unique()

                    for gender in Genders:
                        for geno in Genotypes:
                            for sl in session_length:
                                countKey = f'{strain}_{age}_{gender}_{sl}_{geno}'
                                animal_count[countKey] = np.sum((animalList['Gender'] == gender) & 
                                                                (animalList['Genotype'] == geno) &
                                                                (animalList['Session_length'] == sl))
                                sub_animalList = animalList[(animalList['Gender'] == gender) & 
                                                            (animalList['Genotype'] == geno) & 
                                                            (animalList['Session_length'] == sl)] 

                            count = 0
                            count_key = f'{strain}_{age}_{gender}_{sl}_{geno}_video'
                            for animal in sub_animalList['AnimalID']:
                                if os.path.exists(os.path.join(root_dir, 'Data', str(animal), 'Odor', 'BehavioralRecording')):
                                    count += 1
                            animal_count[count_key] = count

        # save the animal_count in root_folder
        savename = os.path.join(odor_dir, 'animal_count.csv')
        pd.DataFrame.from_dict(animal_count, orient='index').to_csv(savename)


    #%% for odor behavioral data
    # analyze the TSC2 behavioral recording data
    strain = 'TSC2_adol'
    root_dir = os.path.join(r'Y:\HongliWang\Odor', strain)

    Odor = BehDataOdor(root_dir, strain)
    Odor.load_data()
    #Odor.session_analysis()
    #Odor.plot_performance()
    
    #%% model fitting (only implemented policy gradient for now)
    #Odor.model_fitting(fit_mode='session', model_name='policy_gradient')

    Odor.model_fitting(fit_mode='session', model_name='hybrid')
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
        fetch_rotarod(google_url, rotarod_dir, strains, ages)
        
        # preprocess step 2: organize the rotarod videos and DLC files from raw video folders
        # to pipeline folders after clean up
        # move the videos to the corresponding folders in root_dir
        # find the corresponding dlc files from dlc_folder
        # move videos without DLC files to a separate folder for DLC labeling

        video_folder = r'Y:\HongliWang\Rotarod\rawRecordings_260713'
        #dlc_folder = r'Y:\HongliWang\Rotarod\Filtered_DLC'
        dlc_folder = r'Y:\HongliWang\Rotarod\DLC_labeling'
        dlc_labeled_folder = r'Y:\HongliWang\Rotarod\DLC_labeled'
        litpose_labeled_folder = r'Y:\HongliWang\Rotarod\litPose_labeled'
        # remove size 0 files and clean the filenames
        clean_rotarod_videos(video_folder)

        organize_beh_videos(rotarod_dir, video_folder, dlc_folder, litpose_labeled_folder)


    #%% count the number of animals and trials for each strain, age, and genotype
    if count_animals:
        animal_count = {}
        for strain in strains:
            for age in ages:
                root_dir = os.path.join(rotarod_dir, strain + '_' + age)
                strain_folder = strain + '_' + age

                # read the animalLst.csv file
                animalList_file = os.path.join(root_dir,  'Data', 'AnimalList.csv')
                animalList = pd.read_csv(animalList_file)

                if 'Gender' in animalList.columns:
                    Genders = ['M', 'F']
                else:
                    Genders = ['M']
                Genotypes = animalList['Genotype'].unique()

                for gender in Genders:
                    for geno in Genotypes:
                        countKey = f'{strain}_{age}_{gender}_{geno}'
                        if 'F' in Genders:
                            animal_count[countKey] = np.sum((animalList['Gender'] == gender) & (animalList['Genotype'] == geno))
                            sub_animalList = animalList[(animalList['Gender'] == gender) & (animalList['Genotype'] == geno)] 
                        else:
                            animal_count[countKey] = np.sum(animalList['Genotype'] == geno)
                            sub_animalList = animalList[animalList['Genotype'] == geno]

                        count = 0
                        count_key = f'{strain}_{age}_{gender}_{geno}_video'
                        for animal in sub_animalList['AnimalID']:
                            if os.path.exists(os.path.join(root_dir, 'Data', animal)):
                                count += 1
                        animal_count[count_key] = count

        # save the animal_count in root_folder
        savename = os.path.join(rotarod_dir, 'animal_count.csv')
        pd.DataFrame.from_dict(animal_count, orient='index').to_csv(savename)


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
