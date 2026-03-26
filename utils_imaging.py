from datetime import datetime
from os import error
from xml.etree.ElementPath import find
import numpy as np

# helper function to convert ISO timestamp to milliseconds since midnight
def iso_to_timeofday(ts):
    # truncate microseconds to 6 digits for Python
    if '.' in ts:
        main, frac = ts.split('.')
        if '-' in frac or '+' in frac:  # handle timezone offset
            frac, tz = frac[:-6], frac[-6:]
        else:
            tz = ''
        frac = frac[:6]  # keep 6 digits
        ts = main + '.' + frac + tz
    dt = datetime.fromisoformat(ts)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - midnight).total_seconds()


def AI_timeStamp_correction(AI_timestamp):
    # read AI time stamps and correct for jitters in timestamp
    # it should be exactly 1 second between each 1000 timestamps
    sweep_freq = 1000
    diff_ts = np.diff(AI_timestamp)

    #Step 1: finding the index of timestamp that come shorter than 1000ms (probably due to previous delay)
    cond1 = np.concatenate(([0], diff_ts - 1000)) > 1
    cond2 = np.concatenate((diff_ts - 1000, [0])) < -1
    timestamp_shift_target1=np.where(cond1 & cond2)[0]

    # find the condition that a too long sweep is followed by a too short sweep, which indicates a jitter in timestamp
    jitter_flag = ((diff_ts - 1000) > 1).astype(int) - ((diff_ts - 1000) < -1).astype(int)
    jitter_flag = np.concatenate(([0], jitter_flag))

    timestamp_shift_target2 = np.where(np.diff(jitter_flag) == -2)[0]
    # --- check ---
    if np.sum(timestamp_shift_target1 - timestamp_shift_target2) != 0:
        raise ValueError("try again")
    
    Analog_LV_timestamp1 = AI_timestamp.copy()
    for i in timestamp_shift_target1:
        correction = (Analog_LV_timestamp1[i+1] - Analog_LV_timestamp1[i]) - 1000
        Analog_LV_timestamp1[i] = AI_timestamp[i] + correction

    #Step 2: finding the index of timestamp that come shorter than 1000ms (probably due to previous delay 2 seconds ago)
    diff_ts = np.diff(Analog_LV_timestamp1)

    cond1 = np.concatenate(([0], diff_ts - 1000)) > 1
    cond2 = np.concatenate((diff_ts - 1000, [0])) < -1

    timestamp_shift_target1 = np.where(cond1 & cond2)[0]

    for i in timestamp_shift_target1:
        correction = (Analog_LV_timestamp1[i+1] - Analog_LV_timestamp1[i]) - 1000
        Analog_LV_timestamp1[i] = Analog_LV_timestamp1[i] + correction

    # --- Step 2: second pass (2-second correction) ---
    diff_ts = np.diff(Analog_LV_timestamp1)

    cond1 = np.concatenate(([0], diff_ts - 1000)) > 1

    # MATLAB: [(diff(Analog_LV_timestamp1(2:end))-1000); 0; 0]
    diff_shifted = np.diff(Analog_LV_timestamp1[1:]) - 1000
    cond2 = np.concatenate((diff_shifted, [0, 0])) < -1

    timestamp_shift_target2 = np.where(cond1 & cond2)[0]

    Analog_LV_timestamp2 = Analog_LV_timestamp1.copy()

    for i in timestamp_shift_target2:
        # MATLAB: i:2:i+2  → indices [i, i+2]
        correction = (Analog_LV_timestamp2[i+2] - Analog_LV_timestamp2[i]) - 2000

        Analog_LV_timestamp2[i+1] = Analog_LV_timestamp1[i+1] + correction
        Analog_LV_timestamp2[i]   = Analog_LV_timestamp1[i]   + correction

    n = len(AI_timestamp)

    # initialize
    AI_time_interp = np.zeros(n * sweep_freq)

    # assign every 1000th sample (MATLAB: 1000:1000:end)
    AI_time_interp[sweep_freq-1::sweep_freq] = Analog_LV_timestamp2

    # fill backward within each 1000-sample block
    for i in range(n):
        end_idx = i * sweep_freq + sweep_freq - 1  # MATLAB i*sweep_freq → Python index

        AI_time_interp[i*sweep_freq : i*sweep_freq + sweep_freq - 1] = (
            np.arange(-sweep_freq + 1, 0) + AI_time_interp[end_idx]
        )

    return AI_time_interp

def bootstrap(data, dim, dim0, n_sample=1000):
    """
    input:
    data: data matrix for bootstrap
    dim: the dimension for bootstrap, should be data.shape[1]
    dim0: the dimension untouched, shoud be data.shape[0]
    n_sample: number of samples for bootstrap. default: 1000
    output:
    bootRes={'bootAve','bootHigh','bootLow'}
    """
    # Resample the rows of the matrix with replacement
    if len(data)>0:  # if input data is not empty
        bootstrap_indices = np.random.choice(data.shape[dim], size=(n_sample, data.shape[dim]), replace=True)

        # Bootstrap the matrix along the chosen dimension
        bootstrapped_matrix = np.take(data, bootstrap_indices, axis=dim)

        meanBoot = np.nanmean(bootstrapped_matrix,2)
        bootAve = np.nanmean(bootstrapped_matrix, axis=(1, 2))
        bootHigh = np.nanpercentile(meanBoot, 97.5, axis=1)
        bootLow = np.nanpercentile(meanBoot, 2.5, axis=1)

    else:  # return nans
        bootAve = np.full(dim0, np.nan)
        bootLow = np.full(dim0, np.nan)
        bootHigh = np.full(dim0, np.nan)
        # bootstrapped_matrix = np.array([np.nan])

    # bootstrapped_2d = bootstrapped_matrix.reshape(80,-1)
    # need to find a way to output raw bootstrap results
    tempData = {'bootAve': bootAve, 'bootHigh': bootHigh, 'bootLow': bootLow}
    index = np.arange(len(bootAve))
    bootRes = pd.DataFrame(tempData, index)

    return bootRes
