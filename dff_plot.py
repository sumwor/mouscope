import glob
import os
import pickle

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()

import numpy as np

dff_file = r'Y:\HongliWang\Miniscope\ASD\Analysis\ASDC001\Odor\Imaging\20260115\Result\dFF_results.pkl'

with open(dff_file, 'rb') as f:
    dff_results = pickle.load(f)

dff = dff_results['dFF']  # (nFrames, nCells)

# recover the real per-frame time (s) from the aligned imaging timestamp file
# that lives alongside the raw data for this animal/date
path_parts = dff_file.split(os.sep)
animal, date = path_parts[-6], path_parts[-3]
data_root = os.sep.join(path_parts[:-7] + ['Data'])
imaging_folder = os.path.join(data_root, animal, 'Odor', 'Imaging', f'{animal}_{date}')
ts_file = glob.glob(os.path.join(imaging_folder, f'{animal}_*_ImgTimeStamp_*_aligned.csv'))[0]

import pandas as pd
img_timestamp = pd.read_csv(ts_file)
time = img_timestamp['AlignedTimeStamp'].to_numpy()

window_dur = 60  # seconds
time_mask = time <= (time[0] + window_dur)
time = time[time_mask]
dff = dff[time_mask, :]

n_neurons = 10
offset = 10  # dF/F traces have occasional large transients; needs more room than a small offset
colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))

fig, ax = plt.subplots(figsize=(12, 8))
for neuron in range(n_neurons):
    ax.plot(time, dff[:, neuron] + neuron * offset, color=colors[neuron], linewidth=1.8)

ax.axis('off')

# single x/y scale bar instead of axes
time_scale = 5.0    # seconds
dff_scale = 5.0     # dF/F
x0 = time[-1] + 0.02 * (time[-1] - time[0])
y0 = 0

ax.plot([x0, x0 + time_scale], [y0, y0], color='k', linewidth=1.5)
ax.plot([x0, x0], [y0, y0 + dff_scale], color='k', linewidth=1.5)
ax.text(x0 + time_scale / 2, y0, f'{time_scale:g} s', ha='center', va='top')
ax.text(x0, y0 + dff_scale / 2, f'{dff_scale:g} dF/F', ha='right', va='center', rotation=90)

plt.tight_layout()

savefolder = os.path.dirname(dff_file)
savename = os.path.join(savefolder, 'dff_example_traces')
fig.savefig(savename + '.png', dpi=300)
ax.patch.set_visible(False)
fig.savefig(savename + '.svg', transparent=True)

plt.show()
