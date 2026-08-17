# utility functions for computational models
import os
import json

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()

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
    
def fit_hybrid_bias_model(
    data,
    animalID=None,
    savedatapath=None,
    n_starts=100,
    random_seed=0,
    initial_Q=None,
):
    """
    Python equivalent of the MATLAB fitting section:

        fit_hybrid_bias_models()

    Fits:
        a0b1s_hybrid

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain:
            schedule
            actions
            reward

        MATLAB uses:
            schedule
            action
            reward1 > 0

    animalID : optional
        Animal identifier for output.

    n_starts : int
        Number of random starting points for multistart optimization.
        MATLAB GlobalSearch also uses multiple starting points.

    random_seed : int
        Random seed for reproducibility.

    initial_Q : array-like, optional
        Initial Q values. If None, Q starts at 0.5.

    Returns
    -------
    fit_result : dict
    """

    rng = np.random.default_rng(random_seed)


    data = data.copy()

    # Convert missing strings to NaN
    data.replace(["NAN", "NaN", "nan", "None", ""],np.nan,inplace=True)

    data = data.dropna(subset=["schedule", "actions"]).reset_index(drop=True)

    schedules = pd.to_numeric(data["schedule"],errors="coerce").to_numpy(dtype=float)

    actions = pd.to_numeric(data["actions"],errors="coerce").to_numpy(dtype=float)

    rewards = pd.to_numeric(data["reward"], errors="coerce").fillna(0).to_numpy(dtype=float)

    rewards = (rewards > 0).astype(float)

    valid = (np.isfinite(schedules)& np.isfinite(actions))

    schedules = schedules[valid]
    actions = actions[valid]
    rewards = rewards[valid]

    if len(actions) == 0:
        raise ValueError(
            "No valid trials found."
        )


    if not np.all(np.isin(actions, [0, 1])):
        raise ValueError(
            "Python actions must be coded as 0/1."
        )

    choices = actions.astype(int) + 1

    valid = np.isfinite(schedules)

    schedules = schedules[valid]
    choices = choices[valid]
    rewards = rewards[valid]

    stimulus_values = np.sort(np.unique(schedules))

    stim_to_idx = {
        stim: idx
        for idx, stim in enumerate(stimulus_values)
    }

    stim_idx = np.array(
        [
            stim_to_idx[stim]
            for stim in schedules
        ],
        dtype=int,
    )

    n_stimuli = len(stimulus_values)
    n_trials = len(choices)

    bounds = [
        (1e-6, 20.0),   # beta
        (1e-6, 1.0),    # alpha
        (-1.0, 1.0),    # stick
        (1e-6, 1.0),    # lapse
        (1e-6, 1.0),    # ret
        (1e-6, 1.0),    # bias
    ]

    pnames = [
        "beta",
        "alpha",
        "stick",
        "lapse",
        "ret",
        "bias",
    ]


    beta_mu = 5.0
    beta_sigma = 7.0

    def negative_log_posterior(theta):

        nll = a0b1s_hybrid_neg_log_likelihood(
            theta,
            stim_idx,
            choices,
            rewards,
            initial_Q=initial_Q,
        )

        beta = theta[0]

        beta_prior_penalty = (
            (beta - beta_mu) ** 2
            / (2.0 * beta_sigma ** 2)
        )

        return nll + beta_prior_penalty

    best_result = None

    for start_idx in range(n_starts):

        # Random starting point within bounds
        x0 = np.array([
            rng.uniform(low, high)
            for low, high in bounds
        ])


        result = minimize(
            negative_log_posterior,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 2000,
                "ftol": 1e-10,
                "disp": False,
            },
        )


        if (
            best_result is None
            or result.fun < best_result.fun
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Optimization failed for all starting points."
        )


    opt_params = best_result.x

    #beta, alpha, stick, lapse, ret, bias = opt_params

    (
        nll,
        p_right,
        p_correct,
        Q_history,
        state_prob,
        stick_history,
        prediction_error,
    ) = a0b1s_hybrid_neg_log_likelihood(
        opt_params,
        stim_idx,
        choices,
        rewards,
        initial_Q=initial_Q,
        return_latents=True,
    )

    n_params = len(opt_params)

    AIC = 2.0 * nll+ 2.0 * n_params
    BIC = 2.0 * nll+ np.log(n_trials) * n_params
    AIC0 = -2.0* np.log(1.0 / 3.0)* n_trials
    psr2 = (AIC0 - AIC)/ AIC0
    
    # ============================================================
    # Store results
    # ============================================================

    params = {
        name: float(value)
        for name, value
        in zip(pnames, opt_params)
    }

    fit_result = {
        "model": "a0b1s_hybrid",
        "animalID": animalID,
        "params": params,
        "NLL": float(nll),
        # MATLAB optimizes posterior, not just likelihood
        "negative_log_posterior": float(
            best_result.fun
        ),

        "AIC": float(AIC),
        "BIC": float(BIC),
        "AIC0": float(AIC0),
        "psr2": float(psr2),
        "optimizer_success": bool(best_result.success),
        "optimizer_message": str(best_result.message),
        "optimizer_nit": int(best_result.nit),
        "stimulus_values": (stimulus_values.tolist()),
        "pRight_fit": p_right.tolist(),
        "pCorrect_fit": p_correct.tolist(),
        "Q_history": Q_history.tolist(),
        "state_probability": state_prob.tolist(),
        "stickiness": stick_history.tolist(),
        "prediction_error":  prediction_error.tolist(),
    }


    with open(savedatapath, 'w') as f:
        json.dump(
            fit_result,
            f,
            indent=4,
            default= lambda o: o.tolist() if isinstance(o, np.ndarray)
                            else int(o) if isinstance(o, np.integer)
                            else float(o) if isinstance(o, np.floating)
                            else str(o)
        )


    return fit_result

def a0b1s_hybrid_neg_log_likelihood(
    theta,
    stimuli,
    choices,
    rewards,
    initial_Q=None,
    return_latents=False,
):
    """

    Parameters
    ----------
    theta : array-like, length 6
        [beta, alpha, stick, lapse, ret, bias]

    stimuli : array-like
        Stimulus/schedule ID for each trial.

    choices : array-like
        MATLAB choice coding:
            1 = left / A1
            2 = right / A2

    rewards : array-like
        Binary rewards, 0 or 1.

    initial_Q : array-like, shape (n_stimuli, 2), optional
        Initial Q values. If None, initialized to 0.5.

    return_latents : bool
        Whether to return trial-by-trial latent variables.

    Returns
    -------
    nll : float
        Negative log likelihood.

    If return_latents=True:
        nll, p_rl, p_random, p_choice, Q_history,
        state_prob_history, stick_history, prediction_error
    """

    # ------------------------------------------------------------
    # Parameters -- same order as MATLAB
    # ------------------------------------------------------------
    beta, alpha_param, stick_param, lapse, ret, bias = theta

    alpha = np.array([1e-6, alpha_param], dtype=float)

    stick = np.array([0.0, 0.0, stick_param, stick_param],dtype=float)

    # MATLAB:
    # epsilon = 1e-6;
    epsilon = 1e-6

    T = np.array([
        [1.0 - ret, lapse],
        [ret, 1.0 - lapse]
    ])

    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    
    n_trials = len(choices)

    if len(stimuli) != n_trials or len(rewards) != n_trials:
        raise ValueError("stimuli, choices, and rewards must have equal length.")

    unique_stimuli = np.unique(stimuli)
    stim_to_idx = {
        stim: i for i, stim in enumerate(unique_stimuli)
    }

    stim_idx = np.array(
        [stim_to_idx[s] for s in stimuli],
        dtype=int
    )

    n_stimuli = len(unique_stimuli)

    if initial_Q is None:
        Q = np.full((n_stimuli, 2), 0.5, dtype=float)
    else:
        Q = np.asarray(initial_Q, dtype=float).copy()

        if Q.shape != (n_stimuli, 2):
            raise ValueError(
                f"initial_Q must have shape {(n_stimuli, 2)}, "
                f"got {Q.shape}"
            )


    Q_history = np.full(
        (n_trials, n_stimuli, 2),
        np.nan,
        dtype=float
    )

    p_right = np.full(n_trials, np.nan)
    p_correct = np.full(n_trials, np.nan)
    # Probability of the two latent states:
    # state 1 = random
    # state 2 = RL
    state_prob = np.full((n_trials, 2), np.nan)

    stick_history = np.full(n_trials, np.nan)
    prediction_error = np.full(n_trials, np.nan)

    llh = np.log(0.5)
    lt=llh
    p = np.array(
        [lapse, 1.0 - lapse],
        dtype=float
    )

    s = stim_idx[0]

    q_left = Q[s, 0]
    q_right = Q[s, 1]

    z = beta * (q_right - q_left)

    b_right_rl = epsilon / 2.0 + \
        (1.0 - epsilon) / (1.0 + np.exp(-z))

    b = np.array([
        1.0 - b_right_rl,
        b_right_rl
    ])

    # Random-state choice probabilities
    b1 = np.array([1.0 - bias,bias])

    # Store first-trial latent variables
    p_right[0] = (b1[1] * p[0]+ b[1] * p[1])
    if stimuli[0] == 1:
        p_correct[0] = p_right[0]
    else:
        p_correct[0] = 1 - p_right[0]
    state_prob[0] = p
    Q_history[0] = Q
    #lt_list = []
    #lt_list.append(llh)

    # ------------------------------------------------------------
    for k in range(1, n_trials):
        prev_s = stim_idx[k - 1]
        prev_choice = choices[k - 1]
        prev_reward = rewards[k - 1]
        prev_choice_idx = choices[k - 1] - 1
    
        p = (
            b1[prev_choice_idx] * p[0] * T[:, 0]
            + b[prev_choice_idx] * p[1] * T[:, 1]
        ) / np.exp(lt)

        current_choice_idx = choices[k] - 1

        q_before = Q[prev_s, prev_choice_idx]

        prediction_error[k - 1] = prev_reward - q_before

        Q[prev_s, prev_choice_idx] += (
            alpha[int(prev_reward > 0)]
            * (prev_reward - q_before)
        )

        side = np.zeros(4, dtype=float)

        previous_choice_side = 2.0 * (1.5 - prev_choice)

        same_stimulus = (
            stim_idx[k - 1] == stim_idx[k]
        )

        if same_stimulus and prev_reward == 0:
            side[0] = previous_choice_side

        if not same_stimulus and prev_reward == 0:
            side[1] = previous_choice_side

        if same_stimulus and prev_reward > 0:
            side[2] = previous_choice_side

        if not same_stimulus and prev_reward > 0:
            side[3] = previous_choice_side

        stick_effect = np.dot(stick, side)

        stick_history[k] = stick_effect

        # current trial
        current_s = stim_idx[k]

        q_left = Q[current_s, 0]
        q_right = Q[current_s, 1]

        z = beta * (
            q_right
            - q_left
            - stick_effect
        )

        b_right_rl = (
            epsilon / 2.0
            + (1.0 - epsilon) / (1.0 + np.exp(-z))
        )

        b = np.array([
            1.0 - b_right_rl,
            b_right_rl
        ])


        b1[1] = bias
        b1[0] = 1.0 - bias
        current_p_choice = b1[current_choice_idx] * p[0] + b[current_choice_idx] * p[1]

        p_right[k] = (b1[1] * p[0]+ b[1] * p[1])

        lt = np.log(current_p_choice)

        llh += lt
        #lt_list.append(lt)

        # Store

        if stimuli[k] == 1:
            p_correct[k] = p_right[k]
        else:
            p_correct[k] = 1-p_right[k]
        state_prob[k] = p
        Q_history[k] = Q

    nll = -llh

    if return_latents:
        return (
            nll,
            p_right,
            p_correct,
            Q_history,
            state_prob,
            stick_history,
            prediction_error,
        )


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

    elif model_label == 'hybrid_Q':
        q_history = np.asarray(latent_fit["Q_history"], dtype=float)
        q_diff = np.squeeze(np.diff(q_history, axis=1))
        p_engaged = np.asarray(latent_fit["state_probability"], dtype=float)
        bias = float(latent_fit["params"]["bias"])
        stickiness = float(latent_fit["params"]["stick"])

        w_mode = np.vstack([
            q_diff,
            np.full_like(q_diff, bias),
            np.full_like(q_diff, stickiness),
        ])
        weights = ["Q_right_minus_left", "bias", "stickiness"]

        pCorrect_fit_smooth = (
            pd.Series(latent_fit["pCorrect_fit"])
            .rolling(60, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    else:
        raise ValueError(f"Unsupported model_label: {model_label}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
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

    if model_label == 'policy_gradient':
        latent_x = np.arange(1, w_mode.shape[1] + 1)
        for ii in range(w_mode.shape[0]):
            label = weights[ii] if ii < len(weights) else f"weight_{ii + 1}"
            axs[1].plot(latent_x, w_mode[ii], linewidth=3, label=label)
        axs[1].set_ylabel("Latent weight")
        axs[1].set_xlabel("Trial")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].legend(frameon=False)
    elif model_label == 'hybrid_Q':
        latent_x = np.arange(1, q_diff.shape[0] + 1)
        axs[1].plot(latent_x, q_diff[:,0], linewidth=3)
        axs[1].plot(latent_x, q_diff[:,1], linewidth=3)
        axs[1].set_ylabel("Delta Q")
        axs[1].set_xlabel("Trial")
        axs[1].set_ylim([-0.5, 0.5])
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        #axs[1].legend(frameon=False)

        # plot p_engaged
        axs[2].plot(latent_x, p_engaged[:,1], linewidth=3)
        axs[2].set_ylabel("P(engaged)")
        axs[2].set_xlabel("Trial")
        axs[2].set_ylim([0, 1])
        axs[2].spines["top"].set_visible(False)
        axs[2].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(savefigpath+'.png', format="png",dpi=300, bbox_inches="tight")
    fig.savefig(savefigpath+'.svg',  format="svg", bbox_inches="tight")    
    plt.close(fig)
    #return fig
