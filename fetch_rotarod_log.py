# open a google sheet with url, go through the entries, look for newly added animals
# add these animals into the proper RR_result.csv files in the animal

import pandas as pd
import numpy as np
import gspread
from datetime import datetime, timedelta
import time
from gspread.exceptions import APIError
from tqdm import tqdm
import os

google_url = 'https://docs.google.com/spreadsheets/d/1LUcjvEakIwHhLN7hiLypfzwQK5Iod2zubOzcCtDorts/edit?gid=235330200#gid=235330200'

gc = gspread.service_account(filename="credentials/rotarod_reader.json")
sh = gc.open_by_url(google_url)

rotarod_ws = sh.worksheet("rotarod")
df = pd.DataFrame(rotarod_ws.get_all_records())

# forward merge for date, start age


# get some other info from another tab

id_ws = sh.worksheet('ASD IDs (COMPREHENSIVE)')
values = id_ws.get_all_values()
ID_df = pd.DataFrame(values[2:], columns=values[1])

merge_col = ['ROTAROD START AGE', 'ROTAROD START DATE', 
                'ODOR START DATE']
for cc in merge_col:
    ID_df[cc] = ID_df[cc].replace('', np.nan)
    ID_df[cc] = ID_df[cc].ffill()
# go over df, if column is empty, find if there is proper values in ID_df

#%% remember to add digging later

columns2fill = ['age', 'date', 'odor', 'genotype']
columns2look = ['ROTAROD START AGE', 'ROTAROD START DATE', 
                'ODOR START DATE', 'GENOTYPE']
nEntries = df.shape[0]

startRow = 408  # records before this row is disregarded
# due to different exp. protocol

for ee in tqdm(range(408, nEntries)):
    animal_id = df.loc[ee, 'ASD ID']
    ID_idx = np.where(ID_df['ASD ID'] == animal_id)[0][0]

    sheet_row = ee + 2  # IMPORTANT
    for cIdx, col in enumerate(columns2fill):
        if (df.loc[ee, col])=='':  # if the entry is empty
            # try to find the corresponding colomn in ID_df

            # update the corresponding value
            sheet_col = df.columns.get_loc(col) + 1

            if col == 'age':
                if not ID_df.loc[ID_idx, columns2look[cIdx]]=='--':
                    startAge = int(ID_df.loc[ID_idx, columns2look[cIdx]])
                    days2add = int(np.floor((df.loc[ee, 'trial'] - 1) / 3))
                    value = startAge + days2add

            elif col == 'date':
                startDate = ID_df.loc[ID_idx, columns2look[cIdx]]
                if not startDate == '--':
                    dt = datetime.strptime(startDate, '%m/%d/%y')
                    dt2 = dt + timedelta(days=int(np.floor((df.loc[ee, 'trial'] - 1) / 3)))
                    value = int(dt2.strftime('%Y%m%d'))

            elif col == 'odor':
                odordt = ID_df.loc[ID_idx, columns2look[cIdx]]
                if not odordt == '--':
                    dt = datetime.strptime(odordt, '%m/%d/%y')
                    value = int(dt.strftime('%Y%m%d'))
                else:
                    value = 0

            elif col == 'genotype':
                value = ID_df.loc[ID_idx, columns2look[cIdx]]

            df.loc[ee, col] = value
            #rotarod_ws.update_cell(sheet_row, sheet_col, value)

values = [df.columns.tolist()] + df.astype(str).values.tolist()

rotarod_ws.clear()
rotarod_ws.update(values, "A1")

          
rotarod_data_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
strains = ['TSC2', 'Shank3B', 'Nlgn3', 'Chd8', 'Cntnap2', 'Scn2A', 'Syngap1']
ages = ['adol', 'adult']

# go over each folder, update the RR_results.csv 
for ss in strains:
    for aa in ages:
        strain_folder = os.path.join(rotarod_data_dir, f'{ss}_{aa}')
        if not os.path.exists(strain_folder):
            os.makedirs(strain_folder)
        data_folder = os.path.join(strain_folder, 'Data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        RR_file = os.path.join(data_folder, 'RR_results.csv')
        animal_file = os.path.join(data_folder, 'AnimalList.csv')
        
        # get the result of a given strain and age from df

        # for 'age', remove empty column with '' first
        age_col = df['age']
        age_col[df['age']==''] = np.nan

        if aa=='adol':
            df_mask = np.logical_and(df['strain']==ss, age_col<45)
        else:
            df_mask = np.logical_and(df['strain']==ss, age_col>=45)

        df_group = df.loc[df_mask,:]
        
        animals = np.unique(df_group['ASD ID'])
        genotypes = []
        for animal in animals:
            geno = np.unique(df_group['genotype'][df_group['ASD ID']==animal])
            genotypes.append(geno[0])

        # update animalList.csv
        animal_list_df = pd.DataFrame({'AnimalID': animals, 'Genotype': genotypes})
        animal_list_df.to_csv(animal_file, index=False)

        # update RR_Results.csv
        # columns: AnimalID, Genpotype, Age, Weight, Date, Trial, Performance, FBT, Odor, Digging

        # remember to add digging later
        performance_df = pd.DataFrame({'AnimalID': df_group['ASD ID'], 
                                       'Genotype': df_group['genotype'],
                                       'Age': df_group['age'],
                                       'Weight': df_group['weight'],
                                       'Date': df_group['date'],
                                       'Trial': df_group['trial'],
                                       'Performance': df_group['performance'],
                                       'FBT': df_group['FBT'],
                                       'Odor': df_group['odor']
                                      })
        performance_df.to_csv(RR_file, index=False)


## inside each folder, check the files under Videos and Speed
# move back to folders with ASDxxx_xxxxxx
import glob
import re
import shutil

strain_folders = ['Cntnap2_adol', 'TSC2_adol']
for strain_folder in strain_folders:
        data_folder = os.path.join(rotarod_data_dir, strain_folder, 'Data')
        videos_folder = os.path.join(data_folder, 'Videos')
        speed_folder = os.path.join(data_folder, 'Speed')
        timeStamp_csv = glob.glob(os.path.join(videos_folder, '*.csv'))
        # for each speed_csv, group trial 123, 456, 789, 101112 together, find the corresponding video file
        # and video timestamp
        # move these files to a folder under Videos/ASDxxx_xxxxxx
        for speed_file in timeStamp_csv:
            match = re.search(r"(ASD\d+)_(\d{6})_trial(\d+)_", speed_file)

            if match:
                animal_id = match.group(1)
                date = match.group(2)
                trial = int(match.group(3))

            speed2move = []
            video2move = []
            timestame2move = []

            pattern_speed = os.path.join(speed_folder, f"{animal_id}_{date}_trial{trial}*.csv")
            speed2move.extend(glob.glob(pattern_speed))

            pattern_video = os.path.join(videos_folder, f"{animal_id}_{date}_trial{trial}*.avi")
            video2move.extend(glob.glob(pattern_video))

            pattern_timestamp = os.path.join(videos_folder, f"{animal_id}_{date}_trial{trial}*.csv")
            timestame2move.extend(glob.glob(pattern_timestamp))

            dest_folder = os.path.join(videos_folder, f"{animal_id}_{date}")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # move the files to the dest_folder
            for file in speed2move:
                shutil.move(file, dest_folder)

            for file in video2move:
                shutil.move(file, dest_folder)

            for file in timestame2move:
                shutil.move(file, dest_folder)

            


               