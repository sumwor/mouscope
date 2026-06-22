# utility functions for computational models
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import os
import json

import psytrack_learning as psy
from psytrack_learning.getMAP import getMAP
from psytrack_learning.helper.helperFunctions import update_hyper, hyper_to_list
from psytrack_learning.helper.jacHessCheck import compHess, compHess_nolog
from psytrack_learning.helper.invBlkTriDiag import getCredibleInterval
from psytrack_learning.hyperparameter_optimization import evd_lossfun
from psytrack_learning.learning_rules import RewardMax, PredictMax, REINFORCE, REINFORCE_base
from psytrack_learning.simulate_learning import reward_max, predict_max, reinforce, reinforce_base
from psytrack_learning.simulate_learning import simulate_learning

plt.ion()


def fit_policy_gradient(data, animalID, savedatapath):
    # Implementation for fitting policy gradient model

    s_a = data['schedule'].copy(); s_b = data['schedule'].copy()
    s_a[s_a==2] = 0
    s_b[s_b==1] = 0
    s_b[s_b==2] = -1
    correct = data['reward'].copy()
    correct[np.isnan(correct)] = 0
    correct[correct>0] = 1
    answer = (data['schedule'].copy()-1.5)*2
    y = data['actions'].copy()
    dayLength = np.array([1])

    # determine inputs
    t = np.array(data["trial"])
    prior = ((t[1:] - t[:-1]) == 1).astype(int)
    prior = np.hstack(([0], prior))

    # Calculate previous average tone value
    s_avg = (s_a + s_b)/2
    # s_avg = (s_avg - np.mean(s_avg))/np.std(s_avg)
    # s_avg = np.hstack(([0], s_avg))
    # s_avg = s_avg * prior  # for trials without a valid previous trial, set to 0

    # Calculate previous correct answer
    h = correct[:-1].astype(int)   # map from (0,1) to (-1,1)
    h = np.hstack(([0], h))
    h = h * prior  # for trials without a valid previous trial, set to 0

    # Calculate previous choice
    c = y[:-1].astype(int)   # map from (0,1) to (-1,1)
    c = np.hstack(([0], c))
    c = c * prior  # for trials without a valid previous trial, set to 0
    stick_all = np.insert(np.array((y[0:-1]-0.5)*2),0,0) # do not depend on reward
    stick = np.insert(np.array(correct[0:-1])*np.array((y[0:-1]-0.5)*2),0,0)
    inputs = dict(s_a = np.array(s_a)[:, None],
                    s_b = np.array(s_b)[:, None],
                    cBoth = np.array(s_avg)[:, None],
                    stick = stick[:,None],
                    stick_all=stick_all[:,None],
                    h = np.array(h)[:, None],
                    c = np.array(c)[:, None])


    dat = dict(
        subject = animalID,
        inputs = inputs,
        s_a = np.array(s_a),
        s_b = np.array(s_b),
        correct = np.array(correct),
        answer = np.array(answer),
        y =np.array(y),
        dayLength=dayLength,
    )

    seed = 42

    np.random.seed(seed)

    rec_learning_rule = REINFORCE

    # fit to actual data
    #weights = {"bias": 1, "s_a": 1, "s_b": 1}
    weights = {"bias": 1, "cBoth": 1,'stick':1}
    K = np.sum([i for i in weights.values()])

    # save estimated hyper parameters

    est_hyper = {
        'alpha': np.nan,
        'sigma': np.nan,
        'weight': ['bias', 'cBoth', 'stick'],
    }

    # Compute
    hyper_guess = {
        'alpha': [2**-6] * K,
        'sigma': [2**-4] * K,
        'sigInit': [2**4] * K,
        'sigDay': None,
    }

    # Optimizing for both sigma and alpha simultaneously
    optList = ['sigma', 'alpha']

    # List of extra arguments used by evd_lossfun in optimization of evidence
    args = {"optList": optList, "dat": dat, "K": K, "learning_rule": REINFORCE,
            "hyper": hyper_guess, "weights": weights, "update_w": True, "wMode": None,
            "tol": 1e-6, "showOpt": True}

    # Optimization, can also use Nelder-Mead but COBYLA is fastest and pretty reliable
    res = minimize(evd_lossfun, hyper_to_list(hyper_guess, optList, K), args=args, method='COBYLA')
    print("Evidence:", -res.fun, "  ", optList, ": ", res.x)

    opt_hyper = update_hyper(res.x, optList, hyper_guess, K)
    wMode, Hess, logEvd, other = getMAP(dat, opt_hyper, weights, W0=None,
                                        learning_rule=rec_learning_rule, showOpt=0, tol=1e-12)
    wMode = wMode.reshape((K, -1), order="C")

    est_hyper['alpha'] = opt_hyper['alpha']
    est_hyper['sigma'] = opt_hyper['sigma']
    AIC = 2 * K - 2 * logEvd
    BIC = K * np.log(len(dat['y'])) - 2 * logEvd


    # save

    #H, g = compHess_nolog(evd_lossfun, rec_dat['res'].x, 5e-2, {"keywords": hess_args})
    #hyp_std = np.sqrt(np.diag(np.linalg.inv(H)))

    #rec_dat['hyp_std'] = hyp_std
    #%% estimate p_right from the fitted weights
    sti = dat['inputs']['cBoth']
    stick = dat['inputs']['stick']
    sti = np.asarray(sti, dtype=float).ravel()
    stick = np.asarray(stick, dtype=float).ravel()

    x = np.vstack((np.ones(len(sti)), sti, stick))
    
    weighted_sum = np.sum(x * wMode, axis=0)
    # estimate the probability to choose right
    pR = expit(-weighted_sum)

    #%% save the file in json
    rec_dat = {"args": args, 'res': res, 'opt_hyper': opt_hyper, 
               "pR_fit": pR, 'weighted_sum': weighted_sum,
               "wMode": wMode, 'AIC': AIC, 'BIC': BIC,
                 'weight': ['bias', 'stim', 'stick']}

    # recover hyper parameters and std
    hess_args = rec_dat['args'].copy()
    hess_args["wMode"] = rec_dat['wMode'].flatten()
    hess_args["learning_rule"] = hess_args["learning_rule"]

    with open(savedatapath, 'w') as f:
        json.dump(
            rec_dat,
            f,
            indent=4,
            default= lambda o: o.tolist() if isinstance(o, np.ndarray)
                            else int(o) if isinstance(o, np.integer)
                            else float(o) if isinstance(o, np.floating)
                            else str(o)
        )

    return rec_dat


def plot_latent_session(resultdf, latent_fit, model_label,savefigpath):

    
    reward = pd.to_numeric(resultdf["reward"], errors="coerce").fillna(0).to_numpy()
    
    # remove miss trials
    nomiss_trials = ~np.isnan(resultdf['actions'])
    reward = reward[nomiss_trials]
    n_trials = len(reward)
    x_plot = np.arange(1, n_trials + 1)
    rewarded = (reward > 0).astype(float)
    window_size = 60
    running_reward_prob = np.full(n_trials, np.nan)
    
    if n_trials >= window_size:
        csum = np.empty(n_trials + 1, dtype=float)
        csum[0] = 0.0
        np.cumsum(rewarded, out=csum[1:])
        running_reward_prob[:n_trials - window_size + 1] = (
            csum[window_size:] - csum[:-window_size]
        ) / window_size
    pCorrect_data_smooth = pd.Series(running_reward_prob).rolling(60, center=True, min_periods=1).mean().to_numpy()

    if model_label == 'Policy Gradient':
        w_mode = np.asarray(latent_fit["wMode"], dtype=float)
        if w_mode.ndim == 1:
            w_mode = w_mode[None, :]
        weights = latent_fit['weight']
        # # estimate the probability of correct based on the latent weights

        sti = latent_fit['args']['dat']['inputs']['cBoth']
        stick = latent_fit['args']['dat']['inputs']['stick']
        sti = np.asarray(sti, dtype=float).ravel()
        stick = np.asarray(stick, dtype=float).ravel()

        # x = np.vstack((np.ones(len(sti)), sti, stick))
        
        # weighted_sum = np.sum(x * w_mode, axis=0)
        # estimate the probability to choose right
        pR = latent_fit['pR_fit']
        # convert pR to pCorrect_fit
        pCorrect_fit = [1 - pR[i] if sti[i]<0 else pR[i] for i in range(len(pR))]
        pCorrect_fit_smooth = pd.Series(pCorrect_fit).rolling(60, center=True, min_periods=1).mean().to_numpy()

        # calculate the derivative of the fitted weights, looking for peaks
        w_mode_derivative = np.gradient(w_mode, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # first subplot: running reward probability of data and fit
    #axs[0].plot(x, running_reward_prob, color="black", linewidth=2)
    axs[0].plot(x_plot, pCorrect_data_smooth, color="black", linewidth=4, label="Data")
    axs[0].plot(x_plot, pCorrect_fit_smooth, color=[0.7, 0.7, 0.7], linestyle="--", linewidth=4, label="Fit")
    axs[0].set_ylim([0, 1])
    axs[0].legend(frameon=False)
    axs[0].set_ylabel("P(reward)")
    axs[0].set_title("Running reward probability")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    latent_x = np.arange(1, w_mode.shape[1] + 1)
    for ii in range(w_mode.shape[0]):
        label = weights[ii] if ii < len(weights) else f"weight_{ii + 1}"
        axs[1].plot(latent_x, w_mode[ii], linewidth=3, label=label)
    axs[1].set_ylabel("Latent weight")
    axs[1].set_xlabel("Trial")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(savefigpath+'.png', format="png",dpi=300, bbox_inches="tight")
    fig.savefig(savefigpath+'.svg',  format="svg", bbox_inches="tight")    
    plt.close(fig)
    #return fig
