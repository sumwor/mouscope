## todo: 1. go over each subfolder in Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\Data
#  find 
import os
import glob
import shutil

root_path  = r'Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol'
filtered_path = r'Y:\HongliWang\Rotarod\Filtered_DLC'
unfiltered_path  = os.path.join(root_path, 'unfiltered_DLC')
animal_folders = [f for f in os.listdir(os.path.join(root_path, 'Data')) if os.path.isdir(os.path.join(root_path, 'Data', f))]

for aa in animal_folders:
    temp_path = os.path.join(root_path, 'Data', aa, 'Rotarod', 'Behavior')
    data_folder = [f for f in os.listdir(temp_path) if os.path.isdir(os.path.join(temp_path, f))]

    # find existing DLC files
    for dd in data_folder:
        dlc_file_pattern = aa+'*DLC*.csv'
        dlc_files = glob.glob(os.path.join(temp_path, dd, dlc_file_pattern))
        for dlc in dlc_files:
            if ('forMoseq' not in dlc) and ('filtered' not in dlc):
                # find filltered dlc file in another folder
                basename = os.path.basename(dlc)
                fildered_dlc_file = os.path.join(filtered_path, basename[:-4]+'_filtered.csv')
                if os.path.exists(fildered_dlc_file):
                    # move the filtered dlc file to the same folder as the original dlc file
                    new_path = os.path.join(temp_path, dd, os.path.basename(fildered_dlc_file))
                    shutil.move(fildered_dlc_file, new_path)
                    shutil.move(dlc, os.path.join(unfiltered_path, os.path.basename(dlc)))
                    print(f'Moved {fildered_dlc_file} to {new_path}')
                else:
                    print(f'Filtered DLC file not found for {dlc}')