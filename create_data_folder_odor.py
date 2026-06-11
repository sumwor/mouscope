## todo 
# read the animal list for AnimalID, create a folder named by the ID under data_folder
# then create subfolder inside the animal folder 'Odor', then create a 'behavior' folder inside the Odor folder
import os
import pandas as pd

root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic'
strain_name = 'Scn2A'

strain_folder = os.path.join(root_dir, strain_name)
data_folder = os.path.join(strain_folder, 'Data')
animal_list = os.path.join(data_folder, 'AnimalList.csv')

# read the animal list
animalList = pd.read_csv(animal_list)
for animal_id in animalList['AnimalID']:
    animal_folder = os.path.join(data_folder, str(animal_id))
    odor_folder = os.path.join(animal_folder, 'Odor')
    behavior_folder = os.path.join(odor_folder, 'behavior')
    
    # create the folders if they don't exist
    os.makedirs(behavior_folder, exist_ok=True)