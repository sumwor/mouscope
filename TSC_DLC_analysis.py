from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()

#%% for odor behavioral data
# analyze the TSC2 behavioral recording data
# root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\Syngap_Res_adult'
# strain = 'SGR_adult'
# Odor = BehDataOdor(root_dir, strain)
# Odor.load_data()
# #Odor.find_eureka()
# Odor.plot_performance()
# Odor.align_timeStamps()
# Odor.DLC_analysis()

#%% for rotarod data
# for rotarod, need to run the code to update the RR_result from google sheet
# run fetch_rotrarod_log.py

root_dir = r'Y:\HongliWang\Rotarod\ASD_strains\Cntnap2_adol' 
strain = 'Cntnap2_adol'
Rotarod = BehDataRotarod(root_dir, strain)
#Rotarod.plot_performance()
Rotarod.load_data()
Rotarod.plot_performance()

Rotarod.align_timeStamps()

back_keypoints = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
front_keypoints = ['nose', 'left ear', 'right ear', 'left hand', 'right hand']

Rotarod.stride_session(back_keypoints, front_keypoints)

Rotarod.stride_summary()
