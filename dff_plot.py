import os

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import pandas as pd
from scipy.io import loadmat

dff_file = r'Y:\HongliWang\Miniscope\adolescent\11_45_05_success\Miniscope\updated_cnmf.mat'

# CaImAn GUI export: results.C_raw / results.C are (nCells, nFrames) for the
# accepted cells; results.raw holds the same fields for all cells before curation
results = loadmat(dff_file, squeeze_me=True, struct_as_record=False)['results']
trace_type = 'C_raw'  # 'C_raw' = raw CNMF trace, 'C' = denoised trace
dff = getattr(results, trace_type).T  # (nFrames, nCells)

# per-frame time (s) from the Miniscope timeStamps.csv saved next to the .mat
img_timestamp = pd.read_csv(os.path.join(os.path.dirname(dff_file), 'timeStamps.csv'))
time = img_timestamp['Time Stamp (ms)'].to_numpy() / 1000
time = time - time[0]
if len(time) != dff.shape[0]:
    raise ValueError(f'{len(time)} timestamps but {dff.shape[0]} frames in {trace_type}')

window_dur = 60  # seconds
time_mask = time <= (time[0] + window_dur)
time = time[time_mask]
dff = dff[time_mask, :]

n_neurons = 10
offset = 30  # CNMF traces reach ~25 a.u. on large transients
colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))

fig, ax = plt.subplots(figsize=(12, 8))
for neuron in range(n_neurons):
    ax.plot(time, dff[:, neuron] + neuron * offset, color=colors[neuron], linewidth=1.8)

ax.axis('off')

# single x/y scale bar instead of axes
time_scale = 5.0    # seconds
dff_scale = 10.0    # a.u. (CNMF units, not dF/F)
x0 = time[-1] + 0.02 * (time[-1] - time[0])
y0 = 0

ax.plot([x0, x0 + time_scale], [y0, y0], color='k', linewidth=1.5)
ax.plot([x0, x0], [y0, y0 + dff_scale], color='k', linewidth=1.5)
ax.text(x0 + time_scale / 2, y0, f'{time_scale:g} s', ha='center', va='top')
ax.text(x0, y0 + dff_scale / 2, f'{dff_scale:g} a.u.', ha='right', va='center', rotation=90)

plt.tight_layout()

savefolder = os.path.dirname(dff_file)
savename = os.path.join(savefolder, 'dff_example_traces')
fig.savefig(savename + '.png', dpi=300)
ax.patch.set_visible(False)
fig.savefig(savename + '.svg', transparent=True)

plt.show()
