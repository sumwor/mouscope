from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()

#%% for odor behavioral data
# analyze the TSC2 behavioral recording data
root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2'

Odor = BehDataOdor(root_dir)
Odor.load_data()
Odor.plot_performance()
# Odor.align_timeStamps()
# Odor.DLC_analysis()

#%% for rotarod data
root_dir = r'Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol' 
Rotarod = BehDataRotarod(root_dir)
Rotarod.plot_performance()
Rotarod.load_data()