from behavioral_pipeline import *

import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

plt.ion()
# analyze the TSC2 behavioral recording data
root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec'

Odor = BehDataOdor(root_dir)
Odor.load_data()
Odor.align_timeStamps()
Odor.DLC_analysis()