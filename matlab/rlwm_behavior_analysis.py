"""
Python conversion of MATLAB functions for RLWM (Reward-based Learning With Memory) 
behavioral data analysis.

Converted from:
- get_RLWM_EventTimes.m
- extract_behavior_df.m
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import warnings


def get_RLWM_EventTimes(filename):
    """
    Get RLWM event times and metadata from a MATLAB data file.
    
    Parameters
    ----------
    filename : str or dict
        Either a string path to a .mat file or a dictionary containing 'exper' field
    
    Returns
    -------
    dict
        Dictionary containing:
        - RLWM_EventTimes: 3xN array [eventID, eventTime, trial]
        - odor_name: odor names for each trial
        - odor_dur: odor duration for each trial
        - schedule: stimulus schedule for each trial
        - portside: port side schedule for each trial
        - result: result for each trial
        - startTime: absolute start time
    
    Event ID mappings:
        1: center port in
        2: center port out
        3: left port in
        4: left port out
        44: last left port out
        5: right port in
        6: right port out
        66: last right port out
        7.01-7.16: new trial, odor 1-16 ON
        81.0: Correct response, withdraw too early
        81.2: Correct response, 2 drops rewarded
        81.3: Correct response, 3 drops rewarded
        82: False Go (lick), white noise on
        83: Missed to respond
        84: Aborted outcome
        9.01-9.03: Water valve on 1-3 times
    """
    
    warnings.filterwarnings('ignore')
    
    out = {}
    
    # Load data
    if isinstance(filename, str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    elif isinstance(filename, dict):
        if 'exper' not in filename:
            raise ValueError("Dictionary must contain 'exper' field")
        data = filename
    else:
        raise ValueError("filename must be a string path or dictionary with 'exper' field")
    
    if not data:
        print("File not found or empty")
        return out
    
    exper = data['exper']
    
    # Determine which field to use
    has_odor_rlwm = hasattr(exper, 'odor_rlwm') and exper.odor_rlwm is not None
    has_odor_rlwm_automatic = hasattr(exper, 'odor_rlwm_automatic') and exper.odor_rlwm_automatic is not None
    
    if has_odor_rlwm_automatic and not has_odor_rlwm:
        useField = 'odor_rlwm_automatic'
    elif has_odor_rlwm and not has_odor_rlwm_automatic:
        useField = 'odor_rlwm'
    elif has_odor_rlwm and has_odor_rlwm_automatic:
        # Both exist, choose based on CountedTrial
        counted_trial_1 = int(exper.odor_rlwm.param.countedtrial.value)
        counted_trial_2 = int(exper.odor_rlwm_automatic.param.countedtrial.value)
        
        if counted_trial_1 > 0 and counted_trial_2 == 0:
            useField = 'odor_rlwm'
        elif counted_trial_2 > 0 and counted_trial_1 == 0:
            useField = 'odor_rlwm_automatic'
        else:
            useField = 'odor_rlwm_automatic'
    else:
        os.error('no Odor_RLWM session found')
    
    # Extract main data
    trial_events = exper.rpbox.param.trial_events.value
    rlwm_module = getattr(exper, useField)
    
    counted_trial = int(rlwm_module.param.countedtrial.value)
    result = np.array(rlwm_module.param.result.value[:counted_trial])
    portside = np.array(rlwm_module.param.port_side.value[:counted_trial])
    schedule = np.array(rlwm_module.param.schedule.value[:counted_trial])
    odor_channel_schedule = np.array(rlwm_module.param.odorchannel.value[:counted_trial])
    odor_name = np.array(rlwm_module.param.odorname.value[:counted_trial])
    
    stim_param = rlwm_module.param.stimparam.value
    param_string = rlwm_module.param.stimparam.user
    
    # Extract left and right reward ratios
    left_p_idx = np.where(np.array(param_string) == 'left reward ratio')[0][0]
    right_p_idx = np.where(np.array(param_string) == 'right reward ratio')[0][0]
    left_p = np.array([float(x) for x in stim_param[:, left_p_idx]])
    right_p = np.array([float(x) for x in stim_param[:, right_p_idx]])
    
    left_reward_p = left_p[schedule - 1]
    right_reward_p = right_p[schedule - 1]
    
    # Process trials
    rlwm_event_times = []
    valid_trials = np.zeros(counted_trial, dtype=bool)
    kk = 0
    
    for k in range(counted_trial):
        trial_idx = k + 1  # MATLAB uses 1-based indexing
        
        if k == 0:
            tt1 = 0
            try:
                trial_events_k = np.array(rlwm_module.param.trial_events.trial[k])
                if len(trial_events_k.shape) == 1:
                    trial_events_k = trial_events_k.reshape(1, -1)
                
                if result[k] in [1.2, 1.3]:
                    tt2 = trial_events_k[-1, 2]
                else:
                    tt2 = trial_events_k[0, 2] if len(trial_events_k) > 0 else 0
                kk += 1
            except:
                tt2 = 0
        else:
            tt1 = tt2
            try:
                trial_events_k = np.array(rlwm_module.param.trial_events.trial[k])
                if len(trial_events_k.shape) == 1:
                    trial_events_k = trial_events_k.reshape(1, -1)
                
                if len(trial_events_k) > 0:
                    if result[k] in [1.2, 1.3]:
                        tt2 = trial_events_k[-1, 2]
                    else:
                        tt2 = trial_events_k[0, 2] if len(trial_events_k) > 0 else 0
                    kk += 1
                else:
                    # Handle missing trial events
                    if result[k] == 0 and k < counted_trial - 1:
                        tt2 = 0
                    else:
                        raise ValueError(f"No trial events for trial {k}")
            except Exception as e:
                # Skip trials with missing events
                continue
        
        # Get events for current trial
        # time, state, channel
        current_te = trial_events[
            (trial_events[:, 1] > tt1) & (trial_events[:, 1] <= tt2),1:4
        ]
        
        if len(current_te) == 0:
            continue 
        
        # Find ITI events
        c1in_time = current_te[
            (np.isin(current_te[:, 1], [9, 19, 512, 0, 1, 11])) & 
            (np.isin(current_te[:, 2], [1])),0
        ]
        
        # Find odor on time
        delay_odor = int(rlwm_module.param.delayodor.value)
        if delay_odor == 1:
            new_trial_odor_on_time = current_te[
                (np.isin(current_te[:, 1], [2, 12, 22])) & 
                (np.isin(current_te[:, 2], [8])),0
            ]
        else:
            new_trial_odor_on_time = current_te[
                (np.isin(current_te[:, 1], [1, 11, 21])) & 
                (np.isin(current_te[:, 2], [8])),0
            ]
        
        if len(new_trial_odor_on_time) == 0:
            continue
        
        # Extract scalar value from array
        if len(new_trial_odor_on_time) >= 2:
            new_trial_odor_on_time = float(new_trial_odor_on_time[-1])
        else:
            new_trial_odor_on_time = float(new_trial_odor_on_time[0])
        
        valid_trials[k] = True
        
        # ITI events
        iti_te = trial_events[
            (trial_events[:, 1] > tt1) & (trial_events[:, 1] < new_trial_odor_on_time) & 
            np.isin(trial_events[:, 3], [1, 2, 3, 4, 5, 6])
        ][:, [1, 2, 3]]
        
        # Process last poke out
        last_poke_out_mask = np.isin(iti_te[:, 2], [4, 6])
        if np.any(last_poke_out_mask):
            last_idx = np.where(last_poke_out_mask)[0][-1]
            iti_te[last_idx, 2] = iti_te[last_idx, 2] * 10 + iti_te[last_idx, 2]
        
        for row in iti_te:
            rlwm_event_times.append([float(row[2]), float(row[0]), float(kk - 0.5)])
        
        # New trial odor on event
        odor_id = float(odor_channel_schedule[k]) / 100
        rlwm_event_times.append([float(7 + odor_id), float(new_trial_odor_on_time), float(kk)])
        
        # Trial events
        tk_te = trial_events[
            (trial_events[:, 1] > new_trial_odor_on_time) & (trial_events[:, 1] <= tt2) & 
            np.isin(trial_events[:, 3], [1, 2, 3, 4, 5, 6])
        ][:, [1, 2, 3]]
        
        tk_te1 = trial_events[
            (trial_events[:, 1] > new_trial_odor_on_time) & (trial_events[:, 1] <= tt2) & 
            (trial_events[:, 2] == 45) & (trial_events[:, 3] == 8)
        ][:, [1, 2, 3]]
        tk_te1[:, 2] = 9.01
        
        tk_te2 = trial_events[
            (trial_events[:, 1] > new_trial_odor_on_time) & (trial_events[:, 1] <= tt2) & 
            (trial_events[:, 2] == 44) & (trial_events[:, 3] == 8)
        ][:, [1, 2, 3]]
        tk_te2[:, 2] = 9.02
        
        tk_te3 = trial_events[
            (trial_events[:, 1] > new_trial_odor_on_time) & (trial_events[:, 1] <= tt2) & 
            (trial_events[:, 2] == 43) & (trial_events[:, 3] == 8)
        ][:, [1, 2, 3]]
        tk_te3[:, 2] = 9.03
        
        tk_te_combined = np.vstack([tk_te, tk_te1, tk_te2, tk_te3]) if len(tk_te) > 0 or len(tk_te1) > 0 or len(tk_te2) > 0 or len(tk_te3) > 0 else np.empty((0, 3))
        
        if len(tk_te_combined) > 0:
            sort_idx = np.argsort(tk_te_combined[:, 0])
            tk_te_combined = tk_te_combined[sort_idx]
            
            for row in tk_te_combined:
                rlwm_event_times.append([float(row[2]), float(row[0]), float(kk)])
        
        # Outcome event
        rlwm_event_times.append([float(80 + result[k]), float(tt2), float(kk)])
    
    # Convert to array
    if rlwm_event_times:
        rlwm_event_times = np.array(rlwm_event_times, dtype=float).T
    else:
        rlwm_event_times = np.empty((3, 0))
    
    # Filter by valid trials
    out['RLWM_EventTimes'] = rlwm_event_times
    out['odor_name'] = odor_name[valid_trials]
    out['schedule'] = schedule[valid_trials]
    
    # Filter portside with filtered reward parameters
    left_reward_p_filtered = left_reward_p[valid_trials]
    right_reward_p_filtered = right_reward_p[valid_trials]
    portside_filtered = portside[valid_trials].astype(float)
    portside_filtered[(left_reward_p_filtered == -1) & (right_reward_p_filtered == -1)] = -1
    out['portside'] = portside_filtered
    
    out['result'] = result[valid_trials]
    
    # Get start time
    try:
        start_time_val = exper.control.param.trialstart.value
        start_seconds = start_time_val[3] * 3600 + start_time_val[4] * 60 + start_time_val[5]
        out['startTime'] = start_seconds
    except:
        out['startTime'] = 0
    
    # Get odor duration
    try:
        stim_param = rlwm_module.param.stimparam.value
        odor_dur = stim_param[schedule[valid_trials] - 1, 5]
        out['odor_dur'] = np.array([float(x) for x in odor_dur])
    except:
        out['odor_dur'] = np.zeros(np.sum(valid_trials))
    
    return out


def backward_times(dmat, outcome_inds, region_func):
    """
    Helper function to extract event times backward in time from outcome events.
    
    Parameters
    ----------
    dmat : np.ndarray
        Event matrix with columns [eventID, eventTime, trial]
    outcome_inds : np.ndarray
        Indices of outcome events
    region_func : callable
        Function that takes a region and returns boolean mask for selection
    
    Returns
    -------
    np.ndarray
        Array of times for each outcome
    """
    result = np.full(len(outcome_inds), np.nan)
    
    for i, outcome_idx in enumerate(outcome_inds):
        start_idx = 0 if i == 0 else outcome_inds[i - 1]
        end_idx = outcome_idx
        
        region = dmat[start_idx:end_idx + 1, :]
        mask = region_func(region)
        times = region[mask, 1]
        
        if len(times) > 0:
            result[i] = times[-1]
    
    return result


def extract_behavior_df(filename):
    """
    Extract behavioral features from RLWM experimental data.
    
    Parameters
    ----------
    filename : str
        Path to a .mat file containing RLWM experimental data
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing behavioral features for each trial:
        - trial: trial number
        - outcome: trial outcome
        - center_in: center port entry time
        - center_out: center port exit time
        - side_in: side port entry time
        - last_side_out: last side port exit time
        - actions: choice side (3=left, 5=right)
        - reward: water reward amount
        - trial_types: trial type
        - odors: odor identity
        - port_side: scheduled port side
        - schedule: stimulus schedule
        - odor_name: odor name (ASCII)
        - odor_dur: odor duration
        - start_time: session start time
    """
    
    # Load data
    data = get_RLWM_EventTimes(filename)
    
    if not data or len(data.get('RLWM_EventTimes', [])) == 0:
        print("No event data found")
        return pd.DataFrame()
    
    dmat = data['RLWM_EventTimes'].T
    
    # Get basic event time features
    outcome_inds = np.where(dmat[:, 0] > 80)[0]
    n_trials = len(outcome_inds)
    
    result_dict = {}
    result_dict['trial'] = np.arange(1, n_trials + 1)
    result_dict['outcome'] = dmat[outcome_inds, 1]
    
    # Identify odor events
    odor_inds = np.where(np.floor(dmat[:, 0]) == 7)[0]
    
    # Center in times (looking backward from outcome)
    result_dict['center_in'] = backward_times(
        dmat, outcome_inds, 
        lambda region: region[:, 0] == 1
    )
    
    # Center out times
    result_dict['center_out'] = backward_times(
        dmat, outcome_inds,
        lambda region: region[:, 0] == 2
    )
    
    # Side in times
    side_in_times = backward_times(
        dmat, outcome_inds,
        lambda region: np.isin(region[:, 0], [3, 5])
    )
    # Mark as NaN if side_in is before center_in (miss trial)
    side_in_times[side_in_times < result_dict['center_in']] = np.nan
    result_dict['side_in'] = side_in_times
    
    # Last side out times (looking forward)
    last_side_out = np.full(n_trials, np.nan)
    for i in range(n_trials):
        start_idx = outcome_inds[i]
        if i < n_trials - 1:
            end_idx = odor_inds[i + 1] if i + 1 < len(odor_inds) else len(dmat)
        else:
            end_idx = len(dmat)
        
        region = dmat[start_idx:end_idx, :]
        so_times = region[(np.isin(region[:, 0], [44, 66])), 1]
        if len(so_times) > 0:
            last_side_out[i] = so_times[-1]
    
    result_dict['last_side_out'] = last_side_out
    
    # Get task features
    # Actions (choice side)
    trial_sel = np.isin(dmat[:, 1], side_in_times) & (dmat[:, 0] < 80)
    actions = np.full(n_trials, np.nan)
    if np.any(trial_sel):
        choice_trials = dmat[trial_sel, 2].astype(int)
        # Only update actions where choice_trials is within valid range
        valid_idx = (choice_trials - 1 >= 0) & (choice_trials - 1 < n_trials)
        actions[choice_trials[valid_idx] - 1] = (dmat[trial_sel, 0][valid_idx] - 3) / 2
    result_dict['actions'] = actions
    
    # Water rewards
    waters = np.full(n_trials, np.nan)
    water_sel = np.floor(dmat[:, 0]) == 9
    if np.any(water_sel):
        water_given = dmat[water_sel, 2].astype(int)
        # Only update waters where water_given is within valid range
        valid_idx = (water_given - 1 >= 0) & (water_given - 1 < n_trials)
        waters[water_given[valid_idx] - 1] = (dmat[water_sel, 0][valid_idx] % 1) * 100
    result_dict['reward'] = waters
    
    # Trial types
    trial_types_mask = np.floor(dmat[:, 0]) > 80
    if np.any(trial_types_mask):
        trial_types = (dmat[trial_types_mask, 0] % 1) / 10
        result_dict['trial_types'] = trial_types
    else:
        result_dict['trial_types'] = np.full(n_trials, np.nan)
    
    # Odor identity
    odor_mask = np.floor(dmat[:, 0]) == 7
    if np.any(odor_mask):
        odors = (dmat[odor_mask, 0] % 1) * 100
        result_dict['odors'] = odors
    else:
        result_dict['odors'] = np.full(n_trials, np.nan)
    
    # Add metadata
    result_dict['port_side'] = data['portside']
    result_dict['schedule'] = data['schedule']
    result_dict['odor_name'] = data['odor_name']
    result_dict['odor_dur'] = data['odor_dur']
    result_dict['start_time'] = np.full(n_trials, data.get('startTime', 0))
    
    # Create DataFrame
    df = pd.DataFrame(result_dict)
    
    return df


def save_behavior_df(filename, output_csv=None):
    """
    Extract behavioral DataFrame and save as CSV.
    
    Parameters
    ----------
    filename : str
        Path to input .mat file
    output_csv : str, optional
        Path for output CSV file. If None, saves as filename_behavior.csv
    
    Returns
    -------
    pd.DataFrame
        The extracted behavioral DataFrame
    """
    df = extract_behavior_df(filename)
    
    if output_csv is None:
        base_name = os.path.splitext(filename)[0]
        output_csv = f"{base_name}_behavior.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"Saved behavioral DataFrame to {output_csv}")
    
    return df


if __name__ == "__main__":
    matFilePath = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_adol\Data\317\Odor\Behavior\ASD317_20230709_p32_odor_RLWM_AB_100-0_rwdsz3_notgood.mat'
    df = save_behavior_df(matFilePath, output_csv=r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_adol\Analysis\317\Behavior\317_behavior_230709.csv')
