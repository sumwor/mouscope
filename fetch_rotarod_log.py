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

#%% Sort TSC2 adolescent rotarod DLC outputs into analysis-ready folders:
# CSVs are made available to both DLC and MoSeq workflows, labeled videos go
# to DLC_video, and unlabeled videos go to DLCforMoseq.
import os
import shutil

rotarod_data_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
tsc2_adol_folder = os.path.join(rotarod_data_dir, 'TSC2_adol')
rotarod_dlc_folder = os.path.join(tsc2_adol_folder, 'rotarod_DLC')
dlc_csv_folder = os.path.join(tsc2_adol_folder, 'Data', 'DLC')
labeled_video_folder = os.path.join(tsc2_adol_folder, 'DLC_video')
moseq_folder = os.path.join(tsc2_adol_folder, 'DLCforMoseq')

for folder in (dlc_csv_folder, labeled_video_folder, moseq_folder):
    os.makedirs(folder, exist_ok=True)

video_exts = {'.avi', '.mp4', '.mov', '.m4v'}

with os.scandir(rotarod_dlc_folder) as entries:
    for entry in entries:
        if not entry.is_file():
            continue

        basename = entry.name
        stem, ext = os.path.splitext(basename)
        ext = ext.lower()
        lower_stem = stem.lower()

        if ext == '.csv':
            shutil.copy2(entry.path, os.path.join(dlc_csv_folder, basename))
            shutil.move(entry.path, os.path.join(moseq_folder, basename))
        elif ext in video_exts and '_labeled' in lower_stem:
            dest_folder = labeled_video_folder
            shutil.move(entry.path, os.path.join(dest_folder, basename))
        elif ext in video_exts:
            dest_folder = moseq_folder
            shutil.move(entry.path, os.path.join(dest_folder, basename))
        else:
            continue

# %% rearrange data structure
import shutil
import re

rotarod_data_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
tsc2_adol_folder = os.path.join(rotarod_data_dir, 'TSC2_adol')
data_folder = os.path.join(tsc2_adol_folder, 'Data')
animal_session_re = re.compile(r'^(ASD\d+)_(\d{6})$')

if os.path.isdir(data_folder):
    with os.scandir(data_folder) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue

            match = animal_session_re.match(entry.name)
            if not match:
                continue

            animal_id = match.group(1)
            dest_folder = os.path.join(data_folder, animal_id, 'Rotarod', 'Behavior')
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(entry.path, os.path.join(dest_folder, entry.name))

# %% move DLC files into matching behavior session folders
dlc_folder = os.path.join(data_folder, 'DLC')
dlc_file_re = re.compile(r'^(ASD\d+)_trial(1[0-2]|[1-9])(\d{4})-(\d{2})-(\d{2})', re.IGNORECASE)

if os.path.isdir(dlc_folder):
    with os.scandir(dlc_folder) as dlc_entries:
        for entry in dlc_entries:
            if not entry.is_file():
                continue

            match = dlc_file_re.match(entry.name)
            if not match:
                continue

            animal_id, _, year, month, day = match.groups()
            session_id = f'{animal_id}_{year[2:]}{month}{day}'
            dest_folder = os.path.join(data_folder, animal_id, 'Rotarod', 'Behavior', session_id)
            if not os.path.isdir(dest_folder):
                continue

            shutil.move(entry.path, os.path.join(dest_folder, entry.name))

#todo: go through each file under Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\Data
# check if the file name contains xxxxxx(YYMMDD) following ASDxxx, if not, place xxxxxx (seprated by _) 
# the correct date for xxxxxx can be found in the subfolder name ASDxxx_xxxxxx
# so the file name becomes ASDxxx_xxxxxx_trialx....,
# the file stays in the subfolder.
# also delete the files if the size is 0
# %%
