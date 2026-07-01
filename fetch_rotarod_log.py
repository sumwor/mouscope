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
from bisect import bisect_left, bisect_right



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

session_folder_re = re.compile(r'^(ASD\d+)_(\d{6})$', re.IGNORECASE)

for root, dirs, files in os.walk(data_folder):
    if not files:
        continue

    session_match = session_folder_re.match(os.path.basename(root))
    if not session_match:
        continue

    animal_id, session_date = session_match.groups()
    dated_prefix = f'{animal_id}_{session_date}'
    animal_prefix_re = re.compile(rf'^{re.escape(animal_id)}(?!_\d{{6}})', re.IGNORECASE)

    for filename in files:
        path = os.path.join(root, filename)
        try:
            if os.path.getsize(path) == 0:
                os.remove(path)
                continue
        except OSError:
            continue

        if filename.lower().startswith(dated_prefix.lower()):
            continue

        if not animal_prefix_re.match(filename):
            continue

        suffix = filename[len(animal_id):].lstrip('_')
        new_filename = f'{dated_prefix}_{suffix}' if suffix else dated_prefix
        new_path = os.path.join(root, new_filename)
        if new_path != path and not os.path.exists(new_path):
            os.rename(path, new_path)
# %%
base_dir = r'Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\Data'
# for subfolder in os.scandir(base_dir):
#     if not subfolder.is_dir():
#         continue

#     behavior_dir = os.path.join(subfolder.path, 'Rotarod', 'Behavior')
#     behavioral_recording_dir = os.path.join(subfolder.path, 'Rotarod', 'BehavioralRecording')
#     if os.path.isdir(behavior_dir) and not os.path.exists(behavioral_recording_dir):
#         os.rename(behavior_dir, behavioral_recording_dir)

animal_folder_re = re.compile(r'^ASD\d+$', re.IGNORECASE)
session_folder_re = re.compile(r'^(ASD\d+)_(\d{6})$', re.IGNORECASE)

with os.scandir(base_dir) as animal_entries:
    for animal_entry in animal_entries:
        if not animal_entry.is_dir() or not animal_folder_re.match(animal_entry.name):
            continue

        behavioral_recording_dir = os.path.join(
            animal_entry.path, 'Rotarod', 'BehavioralRecording'
        )
        if not os.path.isdir(behavioral_recording_dir):
            continue

        has_entry = False
        session_dirs = []
        with os.scandir(behavioral_recording_dir) as recording_entries:
            for recording_entry in recording_entries:
                has_entry = True
                if recording_entry.is_dir() and session_folder_re.match(recording_entry.name):
                    session_dirs.append((recording_entry.name, recording_entry.path))

        if not has_entry:
            shutil.rmtree(animal_entry.path)
            continue

        for session_name, session_path in session_dirs:
            nested_session_path = os.path.join(session_path, session_name)
            if os.path.isdir(nested_session_path):
                shutil.rmtree(nested_session_path)
