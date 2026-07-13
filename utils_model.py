# utility functions for computational models
import os
import json

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit
import numpy as np
import pandas as pd

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
    
def fit_hybrid(data, animalID, savedatapath):
    """Fit a basic hybrid RL model to trial-by-trial odor behavior.

    The model combines stimulus-specific Q-learning with side bias,
    choice stickiness, and a lapse component.

    P(right) = (1 - lapse) * sigmoid(
        beta * (Q_right - Q_left) + bias + stickiness * previous_choice
    ) + lapse * 0.5
    """
    data = data.copy()
    data.replace({"actions": ["NAN", "NaN", "nan", "None", ""]}, np.nan, inplace=True)
    data.replace({"schedule": ["NAN", "NaN", "nan", "None", ""]}, np.nan, inplace=True)
    data = data.dropna(subset=["schedule"]).reset_index(drop=True)

    actions = pd.to_numeric(data["actions"], errors="coerce").to_numpy(dtype=float)
    schedules = pd.to_numeric(data["schedule"], errors="coerce").to_numpy(dtype=float)
    rewards = pd.to_numeric(data["reward"], errors="coerce").fillna(0).to_numpy(dtype=float)
    rewards = (rewards > 0).astype(float)

    valid = np.isfinite(actions) & np.isfinite(schedules)
    actions = actions[valid].astype(int)
    schedules = schedules[valid].astype(int)
    rewards = rewards[valid]

    if actions.size == 0:
        raise ValueError("No valid trials found for hybrid model fitting.")
    if not np.all(np.isin(actions, [0, 1])):
        raise ValueError("Hybrid model expects actions coded as 0/1.")

    stimulus_values = np.sort(np.unique(schedules))
    stim_to_idx = {stim: idx for idx, stim in enumerate(stimulus_values)}
    stim_idx = np.array([stim_to_idx[stim] for stim in schedules], dtype=int)

    n_stimuli = len(stimulus_values)
    n_trials = len(actions)

    initial = np.array([0.1, 5.0, 0.0, 0.0, 0.02], dtype=float)
    bounds = [
        (1e-4, 0.999),   # alpha
        (1e-3, 50.0),    # beta
        (-10.0, 10.0),   # bias
        (-10.0, 10.0),   # stickiness
        (1e-6, 0.4),     # lapse
    ]

    def negative_log_posterior(params):
        nll = neg_log_likelihood(
            params, actions, rewards, stim_idx, n_stimuli,
        )
        beta = params[1]
        beta_prior_penalty = 0.5 * ((beta - 5.0) / 7.0) ** 2
        return nll + beta_prior_penalty

    res = minimize(negative_log_posterior, initial, method="L-BFGS-B", bounds=bounds)

    opt_params = res.x
    nll, pR, pChoice, q_left, q_right, q_diff, prediction_error = neg_log_likelihood(
        opt_params,
        actions,
        rewards,
        stim_idx,
        n_stimuli,
        return_latents=True,
    )

    param_names = ["alpha", "beta", "bias", "stickiness", "lapse"]
    params = {name: float(value) for name, value in zip(param_names, opt_params)}

    n_params = len(param_names)
    AIC = 2 * n_params + 2 * nll
    BIC = n_params * np.log(n_trials) + 2 * nll

    rec_dat = {
        "model": "hybrid_rl",
        "animalID": animalID,
        "params": params,
        "NLL": float(nll),
        "negative_log_posterior": float(res.fun),
        "beta_prior": {"distribution": "normal", "mean": 5.0, "std": 7.0},
        "AIC": float(AIC),
        "BIC": float(BIC),
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "stimulus_values": stimulus_values.tolist(),
        "n_stimuli": int(n_stimuli),
        "actions": actions.tolist(),
        "schedule": schedules.tolist(),
        "reward": rewards.tolist(),
        "pR_fit": pR.tolist(),
        "pChoice_fit": pChoice.tolist(),
        "Q_left": q_left.tolist(),
        "Q_right": q_right.tolist(),
        "Q_diff": q_diff.tolist(),
        "prediction_error": prediction_error.tolist(),
    }

    with open(savedatapath, "w") as f:
        json.dump(rec_dat, f, indent=4)

    return rec_dat

def neg_log_likelihood(params, actions, rewards, stim_idx, n_stimuli, return_latents=False):
    alpha, beta, bias, stickiness, lapse = params

    Q = np.full((n_stimuli, 2), 0.5, dtype=float)
    prev_choice = 0.0

    n_trials = len(actions)
    pR = np.full(n_trials, np.nan)
    pChoice = np.full(n_trials, np.nan)
    q_left = np.full(n_trials, np.nan)
    q_right = np.full(n_trials, np.nan)
    q_diff = np.full(n_trials, np.nan)
    prediction_error = np.full(n_trials, np.nan)

    eps = 1e-12
    nll = 0.0

    for tt in range(n_trials):
        ss = stim_idx[tt]
        choice = actions[tt]
        reward = rewards[tt]

        q_left[tt] = Q[ss, 0]
        q_right[tt] = Q[ss, 1]
        q_diff[tt] = Q[ss, 1] - Q[ss, 0]

        decision_variable = beta * q_diff[tt] + bias + stickiness * prev_choice
        p_right = (1 - lapse) * expit(decision_variable) + lapse * 0.5
        p_right = np.clip(p_right, eps, 1 - eps)

        pR[tt] = p_right
        pChoice[tt] = p_right if choice == 1 else 1 - p_right
        nll -= np.log(pChoice[tt])

        prediction_error[tt] = reward - Q[ss, choice]
        Q[ss, choice] += alpha * prediction_error[tt]

        prev_choice = (choice - 0.5) * 2

    if return_latents:
        return nll, pR, pChoice, q_left, q_right, q_diff, prediction_error
    return nll

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

    elif model_label == 'Hybrid RL':
        q_diff = np.asarray(latent_fit["Q_diff"], dtype=float)
        bias = float(latent_fit["params"]["bias"])
        stickiness = float(latent_fit["params"]["stickiness"])

        w_mode = np.vstack([
            q_diff,
            np.full_like(q_diff, bias),
            np.full_like(q_diff, stickiness),
        ])
        weights = ["Q_right_minus_left", "bias", "stickiness"]

        pCorrect_fit_smooth = (
            pd.Series(latent_fit["pChoice_fit"])
            .rolling(60, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    else:
        raise ValueError(f"Unsupported model_label: {model_label}")

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
