from datetime import datetime
from os import error
from xml.etree.ElementPath import find
import numpy as np
import h5py
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import rankdata
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import random

import h5py
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

def load_dataset(f, obj):
    """Load HDF5 dataset from matlab, resolving HDF5 object references if present."""
    
    data = obj[()]
    
    # Check if it's an array of object references
    if isinstance(data, np.ndarray) and data.dtype == object:
        # dereference each element
        out = np.empty(data.shape, dtype=object)
        for idx, ref in np.ndenumerate(data):
            if isinstance(ref, h5py.Reference):
                out[idx] = f[ref][()]
            else:
                out[idx] = ref
        return out
    
    return data

# @njit
# def auROC_supT(data, labels):
#     # auROC function designed for supT test
#     # every function passed to supT_stats need to take the same data
#     # data: a vector of dFF of cell c, at time t for all trials
#     # label: dependent variables
#     # use numba to speed it up
#     mask = ~np.isnan(data)
    
#     y = labels[mask]
#     x= data[mask]
#     #  skip if not enough data
#     if np.sum(mask) < 2 or len(np.unique(y)) < 2:
#         return np.nan
#     ranks = rankdata(x)

#     n1 = np.sum(y==1)
#     n0 = np.sum(y==0)
#     rank_sum = np.sum(ranks[y==1])  # ranks start at 1
#     auc = (rank_sum - n1*(n1+1)/2) / (n1*n0)
#     return auc

@njit
def rankdata_argsort(x):
    n = len(x)
    order = np.argsort(x)

    ranks = np.empty(n)
    i = 0

    while i < n:
        j = i

        # find tie block
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1

        # average rank for ties
        rank = (i + j) / 2.0 + 1.0

        for k in range(i, j + 1):
            ranks[order[k]] = rank

        i = j + 1

    return ranks


@njit
def auROC_supT(data, labels):

    # count valid
    n = 0
    for i in range(len(data)):
        if not np.isnan(data[i]):
            n += 1

    if n < 2:
        return np.nan

    x = np.empty(n)
    y = np.empty(n)

    k = 0
    for i in range(len(data)):
        if not np.isnan(data[i]):
            x[k] = data[i]
            y[k] = labels[i]
            k += 1

    n1 = 0
    n0 = 0
    for i in range(n):
        if y[i] == 1:
            n1 += 1
        else:
            n0 += 1

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = rankdata_argsort(x)

    rank_sum = 0.0
    for i in range(n):
        if y[i] == 1:
            rank_sum += ranks[i]

    auc = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    return auc


def linear_regr_supT(data, labels):
    """
    Parameters
    ----------
    data : array-like, shape (n_trials,)
        Neural response (e.g., dF/F) for a single cell across trials.
    labels : array-like, shape (n_trials, n_predictors)
        Predictor matrix for linear regression (Sn, Cn, interactions, etc.)

    Returns
    -------
    results : statsmodels regression results object
        Can access .params (coefficients), .pvalues, .tvalues, etc.
        # return coefficient and pvalue only
    """
    # make sure inputs are arrays
    data = np.asarray(data)
    labels = np.asarray(labels)
    
    # remove trials with NaN in either data or labels
    valid_mask = ~np.isnan(data) & ~np.any(np.isnan(labels), axis=1)
    y = data[valid_mask]
    X = labels[valid_mask, :]
    
    # add intercept
    X = sm.add_constant(X)
    
    # fit linear regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    res = {}
    res['coeff'] = results.params
    res['pvalue'] = results.pvalues

    return res

def metric_parallel(data, labels, metric_func):
    # parallel the supT test
    # 
    nTime, nTrials, nCells = data.shape
    
    if 'auROC' in metric_func.__name__:
        # shuffle labels
        trial_type1_mask = (labels['target']==0) & (labels['stratify']==0)  # A left
        trial_type2_mask = (labels['target']==0) & (labels['stratify']==1)  # A right
        trial_type3_mask = (labels['target']==1) & (labels['stratify']==0)  # B left
        trial_type4_mask = (labels['target']==1) & (labels['stratify']==1)  # B right
  
        idx1 = np.where(trial_type1_mask)[0]
        idx2 = np.where(trial_type2_mask)[0]
        idx3 = np.where(trial_type3_mask)[0]
        idx4 = np.where(trial_type4_mask)[0]

        min_trials = min(len(idx1), len(idx2), len(idx3), len(idx4))

        rng = np.random.default_rng()

        s1 = rng.choice(idx1, min_trials, replace=False)
        s2 = rng.choice(idx2, min_trials, replace=False)
        s3 = rng.choice(idx3, min_trials, replace=False)
        s4 = rng.choice(idx4, min_trials, replace=False)
        sel_idx = np.concatenate([s1, s2, s3, s4])

        labels_use = labels['target'][sel_idx]
        labels_sh = np.random.permutation(labels_use)
        metric_sh_all = np.full((nTime, nCells), np.nan)
        supT_shuffle = np.zeros(nCells)
        for cell in range(nCells):
            metric_sh = np.full(nTime, np.nan)
            for t in range(nTime):
                metric_sh_all[t, cell] = metric_func(data[t, sel_idx, cell], labels_sh)
            # supT = max absolute deviation across time
            supT_shuffle[cell] = np.nanmax(np.abs(metric_sh_all[:, cell] - 0.5))
    
    elif 'linear_regr' in metric_func.__name__:
        # shuffle data
        idx = np.random.permutation(labels.shape[0])
        labels_sh = labels[idx, :]# shuffle trials
        n_pred = labels.shape[1]
        metric_sh_all = np.full((nTime, nCells, n_pred+1), np.nan)
        supT_shuffle = np.zeros((nCells,n_pred+1))
        for cell in range(nCells):
            #metric_sh = np.full(nTime, np.nan)
            for t in range(nTime):
                result = metric_func(data[t, :, cell], labels_sh)
                metric_sh_all[t, cell,:] = result['coeff']  # exclude intercept
            # supT = max absolute deviation across time
            supT_shuffle[cell,:] = np.nanmax(np.abs(metric_sh_all[:,cell,:]),0)  # max across time, for each predictor

    return supT_shuffle, metric_sh_all

def supT_stats(data,labels, metric_func, nShuffles = 1000):
    # supT stats to determine signficance for time series
    # label : label to be shuffled
    # metric_func: the function to calculate the metrics that need to be tested (e.g. auROC)
    nTime, nTrials, nCells = data.shape

    # observed metric
    #%% auROR analysis
    if 'auROC' in metric_func.__name__:
        # for auROC, we stratify the different trial types (A left, A right, B left, B right should have the same numbers)
        trial_type1_mask = (labels['target']==0) & (labels['stratify']==0)  # A left
        trial_type2_mask = (labels['target']==0) & (labels['stratify']==1)  # A right
        trial_type3_mask = (labels['target']==1) & (labels['stratify']==0)  # B left
        trial_type4_mask = (labels['target']==1) & (labels['stratify']==1)  # B right
  
        idx1 = np.where(trial_type1_mask)[0]
        idx2 = np.where(trial_type2_mask)[0]
        idx3 = np.where(trial_type3_mask)[0]
        idx4 = np.where(trial_type4_mask)[0]

        min_trials = min(len(idx1), len(idx2), len(idx3), len(idx4))

        maxRepeats = 100
        metric_obs = np.full((nTime, nCells), np.nan)
        #converge_iter = np.full((nTime, nCells), np.nan)

        rng = np.random.default_rng(42)

        for cell in tqdm(range(nCells)):
            for t in range(nTime):
                auc_list = []

                for r in range(maxRepeats):
                    # Balanced subsampling
                    s1 = rng.choice(idx1, min_trials, replace=False)
                    s2 = rng.choice(idx2, min_trials, replace=False)
                    s3 = rng.choice(idx3, min_trials, replace=False)
                    s4 = rng.choice(idx4, min_trials, replace=False)
                    sel_idx = np.concatenate([s1, s2, s3, s4])

                    # Compute metric
                    auc = metric_func(data[t, sel_idx, cell], labels['target'][sel_idx])
                    auc_list.append(auc)

                    # Check convergence after enough repeats
 

                metric_obs[t, cell] = np.mean(auc_list)
                #converge_iter[t, cell] = maxRepeats

        # show the maximum convergence iteration for all cells and time points
        #print(f'Maximum convergence iteration: {np.nanmax(converge_iter)}')
        supT_obs = np.nanmax(np.abs(metric_obs - 0.5), axis=0)

        print('Running supT test...')

        # permutation null distribution, paralleled  
        with Parallel(n_jobs=12, backend="loky") as parallel:
            results = parallel(
                delayed(metric_parallel)(data, labels, metric_func)
                for _ in tqdm(range(nShuffles))
            )

        supT_null = np.zeros((nShuffles, nCells))
        metric_null = np.zeros((nShuffles, nTime, nCells))

        for i, res in enumerate(results):
            supT_null[i, :] = res[0]
            metric_null[i, :, :, ] = res[1]  
 # metric_sh: nTime x nCells

        # p-value: fraction of shuffled supT >= observed supT
        p_values = np.array([np.mean(supT_null[:, cell] >= supT_obs[cell]) 
                            for cell in range(nCells)])
        _, p_adjusted, _ , _ = multipletests(p_values, method='fdr_bh')

    #%% linear regression analysis
    elif 'linear_regr' in metric_func.__name__:
        n_pred = labels.shape[1]
        metric_obs = {}
        metric_obs['coeff'] = np.full((nTime,nCells,n_pred+1), np.nan)
        metric_obs['pvalue'] = np.full((nTime,nCells,n_pred+1), np.nan)
        for cell in range(nCells):
            for t in range(nTime):
                result = metric_func(data[t, :, cell], labels)
        
                metric_obs['coeff'][t, cell, :] = result['coeff']
                metric_obs['pvalue'][t, cell, :] = result['pvalue']
    # plot for coefficients
        # pred = ['stimulus n', 'choice n', 'outcome n', 'stimulus n-1', 'choice n-1', 'outcome n-1']
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # for p in range(n_pred):
        #     plt.subplot(2, 3, p+1)
        #     coeff = np.sum(metric_obs['pvalue'][:, :, p+1]<0.01,1)/metric_obs['pvalue'].shape[1] # skip intercept
        #     plt.plot(coeff)
        #     plt.title(pred[p])
        #     plt.xlabel('Time')
        #     plt.ylabel('Cells')

        supT_obs = np.nanmax(np.abs(metric_obs['coeff']), axis=0)

        # print('Running supT test...')

        # permutation null distribution, paralleled  
        # shuffle dependent variables (data) for linear regression supT test        
        results = Parallel(n_jobs=-1)(
                delayed(metric_parallel)(data, labels, metric_func)
                for _ in tqdm(range(nShuffles))
            )
        supT_null = np.zeros((nShuffles, nCells, n_pred+1))
        metric_null = np.zeros((nShuffles, nTime, nCells, n_pred+1))

        for i, res in enumerate(results):
            supT_null[i, :] = res[0]
            metric_null[i, :, :, :] = res[1]  # metric_sh: nTime x nCells

        # p-value: fraction of shuffled supT >= observed supT
        p_values = (
                np.sum(supT_null >= supT_obs[None, :, :], axis=0) + 1
            ) / (nShuffles + 1)
        # threshold

        p_adjusted = np.full_like(p_values, np.nan)
        for pp in range(n_pred+1):
             _, p_adjusted[:, pp], _ , _ = multipletests(p_values[:, pp], method='fdr_bh')

    #multiple comparison correction

    return metric_obs, p_adjusted, metric_null

def orthogonalize(x1, x2):
    S = x1.reshape(-1, 1)
    C = x2.reshape(-1, 1)
    SC = S * C

    def proj(u, v):
        return (u.T @ v) / (u.T @ u) * u

    S_t = S
    C_t = C - proj(S_t, C)
    SC_t = SC - proj(S_t, SC) - proj(C_t, SC)

    return S_t.ravel(), C_t.ravel(), SC_t.ravel()

def build_time_resolved_predictor(event_times, nTrials, kernel_len, time_bins, trial_duration=0.1):
    """
    Build a time-resolved binary predictor matrix for one event type (stimulus/choice/outcome).

    Parameters
    ----------
    event_times : array of shape (nTrials,)
        The time (relative to trial start) when the event happens for each trial.
    nTrials : int
        Number of trials.
    kernel_len : int
        Number of time bins for the kernel (e.g., 40 bins for -2 to 2 s).
    time_bins : array of shape (kernel_len,)
        Times relative to alignment point for each bin (e.g., np.arange(-2, 2, 0.1))
    trial_duration : float
        Duration of each time bin.

    Returns
    -------
    X : array of shape (nTrials, kernel_len)
        Binary predictor matrix: 1 at the bin when event happens, 0 otherwise.
    """
    X = np.zeros((nTrials, kernel_len))
    for tr in range(nTrials):
        t_event = event_times[tr]
        # find the bin index where event occurs
        bin_idx = np.argmin(np.abs(time_bins - t_event))
        X[tr, bin_idx] = 1
    return X

def d_prime(data):

    # given data with two columns (condition 1 and condition 2), calculate d 
    # input data: stimulus normalized to 1 and 2
    # actions normalized to 0 and 1 (stimulus 1 -> action 1; stimulus 2 -> action 2)
    stim = data['stimulus']   # 1 = A, 2 = B
    choice = data['actions']  # 1 = Left, 0 = Right (adjust if needed)

    # hits: correct "A" responses
    hits = np.sum((stim == 1) & (choice == 1))
    nA   = np.sum(stim == 1)

    # false alarms: incorrectly choosing A on B trials
    fas  = np.sum((stim == 2) & (choice == 1))
    nB   = np.sum(stim == 2)

    hit_rate = hits / nA
    fa_rate  = fas / nB

    # loglinear correction
    hit_rate = (hits + 0.5) / (nA + 1)
    fa_rate  = (fas + 0.5) / (nB + 1)

    dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    return dprime, criterion

def run_decoder(input_x, input_y, stratify_var,classifier, n_shuffle):
    # return a trained decoder that can be used to decode subset of signals in a manner of testing set
    # input_x: nTrials x nNeurons (df/f)
    # input_y: nTrials (labels for each trial. Either stimulus or choice)


    y = input_y.values
    stratify_var = stratify_var.values

    # remove NaNs from stratify_var and df/f
    mask_valid = ~np.isnan(stratify_var)

    mask_valid_x = ~np.isnan(input_x).any(axis=(0, 2)) 

    X_valid = input_x[:, mask_valid, :]       # trials × neurons
    y_valid = y[mask_valid]
    stratify_valid = stratify_var[mask_valid]
    # split into train and test set, with trials balanced for different trial types (e.g. A left, A right, B left, B right should have the same numbers in train and test set)
    nTrials = len(y_valid)

    trial_indices = np.arange(nTrials)

    # get train/test indices using stratification
    train_idx, test_idx = train_test_split(
        trial_indices,
        test_size=0.3,
        stratify=stratify_valid,
        random_state=42
    )

    # get sample weight 
    unique_classes, counts = np.unique(stratify_valid, return_counts=True)
    class_weights = {c: 1.0/count for c, count in zip(unique_classes, counts)}

    # Assign weight to each trial in the training set
    sample_weights = np.array([class_weights[c] for c in stratify_valid])


    X_train = X_valid[:,train_idx, :]
    y_train = y_valid[train_idx]
    X_test = X_valid[:,test_idx, :]
    y_test = y_valid[test_idx]
    sw_train = sample_weights[train_idx]
    #sw_test = sample_weights[test_idx]

    if classifier == 'RandomForest':

        rfc = RFC()
        n_estimators = [int(w) for w in np.linspace(start = 10, stop = 500, num=10)]
        max_depth = [int(w) for w in np.linspace(5, 20, num=10)] # from sqrt(n) - n/2
        min_samples_leaf = [0.1]
        max_depth.append(None)

        # create random grid
        random_grid = {
            'n_estimators': n_estimators,
            'min_samples_leaf': min_samples_leaf,
            'max_depth': max_depth
        }
        rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=False,
                                        random_state=42, n_jobs=-1)
        #rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
        #                                n_iter=100, cv=5, verbose=False,
        #                                random_state=42)

        # Fit the model
        rfc_random.fit(X_train, y_train)
        # print results
        #print(rfc_random.best_params_)

        best_n_estimators = rfc_random.best_params_['n_estimators']
        #best_n_estimators = 10
        #best_max_features = rfc_random.best_params_['max_features']
        best_max_depth = rfc_random.best_params_['max_depth']
        best_min_samples_leaf = rfc_random.best_params_['min_samples_leaf']
        model = RFC(n_estimators = best_n_estimators,
                        max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf, class_weight='balanced')

        # get control value by shuffling trials

        model.fit(X_train, y_train)

        # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
        #from sklearn.metrics import balanced_accuracy_score
        # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
        # calculate decoding accuracy based on confusion matrix
        best_predict = model.predict(X_test)
        proba_estimates = model.predict_proba(X_test)
        pred = confusion_matrix(y_test, best_predict)
        pred_accuracy = np.trace(pred)/np.sum(pred)

        # feature importance
        importance = model.feature_importances_
        # need to return: classfier (to decode specific trial type later)
        #                 shuffled accuracy (control)
        #                 accuracy (decoding results)
        #                 importance
        #                 best parameters of the randomforest decoder


        # control
        pred_shuffle = np.zeros(n_shuffle)
        for ii in range(n_shuffle):
            xInd = np.arange(len(y_test))
            X_test_shuffle = np.zeros((X_test.shape))
            for cc in range(X_train.shape[1]):
                np.random.shuffle(xInd)
                X_test_shuffle[:,cc] = X_test[xInd,cc]

            predict_shuffle = model.predict(X_test_shuffle)
            pred = confusion_matrix(y_test, predict_shuffle)
            pred_shuffle[ii] = np.trace(pred) / np.sum(pred)

    elif classifier == 'SVC':


        # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
        # from sklearn.metrics import balanced_accuracy_score
        # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
        # calculate decoding accuracy based on confusion matrix

        # cross validation within training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        nTime = X_valid.shape[0]
        cv_scores_time = np.zeros(nTime)
        test_scores_time = np.zeros(nTime)
        importance = np.zeros((nTime, X_valid.shape[2]))  
        pred_shuffle = np.zeros((n_shuffle, nTime))
        # nTime x nNeurons
        for t in range(nTime):
            X_train_t = X_train[t, :, :]  # (nTrainTrials, nNeurons)
            X_test_t  = X_test[t, :, :]
            
            # cross-validate on training set
            cv_scores = []
            for tr_idx, val_idx in cv.split(X_train_t, y_train):
                model = SVC(kernel='linear')
                model.fit(X_train_t[tr_idx], y_train[tr_idx], sample_weight=sw_train[tr_idx])
                cv_scores.append(model.score(X_train_t[val_idx], y_train[val_idx]))
            
            cv_scores_time[t] = np.mean(cv_scores)
            
            # Fit final model on full training set and evaluate on test set
            final_model = SVC(kernel='linear')
            final_model.fit(X_train_t, y_train, sample_weight=sw_train)
            test_scores_time[t] = final_model.score(X_test_t, y_test)
            importance[t, :] = final_model.coef_[0]  # feature importance from linear SVM

            # parallel the shuffling
            for sh in range(n_shuffle):
                xInd = np.arange(len(y_test))
                np.random.shuffle(xInd)                # shuffle trials globally
                X_test_shuffle = X_test[t,xInd, :]     
            
                pred_shuffle[sh, t] = final_model.score(X_test_shuffle, y_test)



    decoder = {}
    decoder['classifier'] = model
    decoder['ctrl'] = pred_shuffle
    decoder['accuracy'] = test_scores_time
    decoder['importance'] = importance
    if classifier == 'RandomForest':
        decoder['params'] = rfc_random.best_params_
        decoder['confidence'] = np.mean(proba_estimates,0)
    return decoder


def get_train_test(X, y, test_size, random_state):
    # check number of classes
    random.seed(random_state)
    classes = np.unique(y)
    nClass = len(np.unique(y))

    instance_class = np.zeros(nClass)
    for cc in range(nClass):
        instance_class[cc] = np.sum(y==classes[cc])

    minIns = np.min(instance_class)
    minInd = np.unravel_index(np.argmin(instance_class),instance_class.shape)
    minClass = classes[minInd]

    # split the trials based on test_size and the class with minimum instances
    classCountTest = np.sum(y==minClass)*test_size
    trainInd = []
    testInd = []
    for nn in range(nClass):
        tempClassInd = np.arange(len(y))[y==classes[nn]]
        tempTestInd = random.choices(tempClassInd,
                                        k=int(classCountTest))
        IndRemain = np.setdiff1d(tempClassInd,tempTestInd)
        tempTrainInd = random.choices(IndRemain,
                                        k=int(classCountTest))
        testInd = np.concatenate([testInd,tempTestInd])
        trainInd = np.concatenate([trainInd,tempTrainInd])

    trainInd = trainInd.astype(int)
    testInd = testInd.astype(int)

    X_train = X[trainInd,:]
    X_test = X[testInd,:]
    y_train = y[trainInd]
    y_test = y[testInd]

    return X_train,y_train, X_test, y_test
# %%


def load_h5_item(f, item):
    if isinstance(item, h5py.Dataset):
        data = item[()]
        
        # array of references (MATLAB cell)
        if isinstance(data, np.ndarray) and data.dtype == object:
            return [load_h5_item(f, f[ref]) for ref in data.flat]
        
        # single reference
        if isinstance(data, h5py.h5r.Reference):
            return load_h5_item(f, f[data])
        
        return data

    elif isinstance(item, h5py.Group):
        return {key: load_h5_item(f, item[key]) for key in item.keys()}

    elif isinstance(item, h5py.h5r.Reference):
        return load_h5_item(f, f[item])

    else:
        return item