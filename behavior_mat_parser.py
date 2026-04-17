from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class RLWMParsed:
    rlwm_event_times: np.ndarray
    odor_name: np.ndarray
    odor_dur: np.ndarray
    schedule: np.ndarray
    portside: np.ndarray
    result: np.ndarray
    protocol_name: str


def _mat_struct_to_dict(obj: Any) -> Any:
    """
    Recursively convert scipy loadmat mat_struct-like objects into Python dict/list/scalars.
    Preserve nested shape for MATLAB cell arrays instead of flattening them.
    """
    if hasattr(obj, "_fieldnames"):
        out = {}
        for name in obj._fieldnames:
            out[name] = _mat_struct_to_dict(getattr(obj, name))
        return out

    if isinstance(obj, np.ndarray):
        obj = np.squeeze(obj)

        # MATLAB cell arrays / object arrays
        if obj.dtype == object:
            if obj.ndim == 0:
                return _mat_struct_to_dict(obj.item())
            return _mat_struct_to_dict(obj.tolist())

        # normal numeric arrays
        return obj

    if isinstance(obj, list):
        return [_mat_struct_to_dict(x) for x in obj]

    return obj


def load_exper_file(filename: str) -> Dict[str, Any]:
    """
    Load a MATLAB exper file and convert it to nested Python dicts/arrays.
    """
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    clean = {}
    for k, v in data.items():
        if k.startswith("__"):
            continue
        clean[k] = _mat_struct_to_dict(v)
    return clean


def detect_rlwm_protocol(exper_data: Dict[str, Any]) -> str:
    """
    Return 'odor_rlwm' or 'odor_rlwm_automatic'.
    """
    exper = exper_data.get("exper", {})
    if "odor_rlwm" in exper:
        return "odor_rlwm"
    if "odor_rlwm_automatic" in exper:
        return "odor_rlwm_automatic"
    raise ValueError("No odor_rlwm or odor_rlwm_automatic session found in exper file.")


def _to_1d_numeric(x: Any, dtype=float) -> np.ndarray:
    arr = np.array(x)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = np.array([arr])
    return arr.astype(dtype)


def _extract_protocol_block(exper_data: Dict[str, Any], protocol_name: str) -> Dict[str, Any]:
    exper = exper_data["exper"]
    return exper[protocol_name]["param"]


def _ensure_2d_numeric(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=float)
    arr = np.squeeze(arr)

    if arr.size == 0:
        return np.empty((0, 0), dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def _looks_like_trial_event_matrix(x: Any) -> bool:
    try:
        arr = np.array(x, dtype=float)
        arr = np.squeeze(arr)
        if arr.size == 0:
            return True
        if arr.ndim == 1:
            return arr.shape[0] == 3
        return arr.ndim == 2 and arr.shape[1] == 3
    except Exception:
        return False


def _collect_trial_event_matrices(obj: Any) -> List[np.ndarray]:
    """
    Normalize MATLAB cell arrays of per-trial event matrices into a Python list of (n, 3) arrays.
    """
    mats: List[np.ndarray] = []

    if _looks_like_trial_event_matrix(obj):
        mats.append(_ensure_2d_numeric(obj))
        return mats

    if isinstance(obj, list):
        for item in obj:
            mats.extend(_collect_trial_event_matrices(item))
        return mats

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            mats.extend(_collect_trial_event_matrices(item))
        return mats

    return mats


def _flatten_strings(x: Any) -> List[str]:
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            out.extend(_flatten_strings(item))
        return out
    return [str(x)]


def _backward_times(dmat: np.ndarray, outcome_inds: np.ndarray, event_selector) -> np.ndarray:
    result = np.full(len(outcome_inds), np.nan, dtype=float)
    for i in range(len(outcome_inds)):
        start_ind = 0 if i == 0 else outcome_inds[i - 1]
        end_ind = outcome_inds[i]
        region = dmat[start_ind : end_ind + 1, :]
        mask = event_selector(region)
        itime = region[mask, 1]
        if len(itime) > 0:
            result[i] = itime[-1]
    return result


def get_rlwm_event_times_py(filename: str) -> RLWMParsed:
    """
    Python port of get_RLWM_EventTimes / get_RLWM_automatic_EventTimes.
    """
    data = load_exper_file(filename)
    protocol_name = detect_rlwm_protocol(data)
    param = _extract_protocol_block(data, protocol_name)

    counted_trial = int(np.squeeze(param["countedtrial"]["value"]))
    result = _to_1d_numeric(param["result"]["value"], dtype=float)[:counted_trial]
    schedule = _to_1d_numeric(param["schedule"]["value"], dtype=float)[:counted_trial]
    portside = _to_1d_numeric(param["port_side"]["value"], dtype=float)[:counted_trial]
    odor_name = _to_1d_numeric(param["odorname"]["value"], dtype=float)[:counted_trial]
    odor_channel = _to_1d_numeric(param["odorchannel"]["value"], dtype=float)[:counted_trial]

    stimparam = param["stimparam"]["value"]
    stimparam_user = _flatten_strings(param["stimparam"]["user"])

    odor_dur = np.array([float(stimparam[int(s) - 1][5]) for s in schedule], dtype=float)

    left_idx = stimparam_user.index("left reward ratio")
    right_idx = stimparam_user.index("right reward ratio")

    left_all = np.array([float(row[left_idx]) for row in stimparam], dtype=float)
    right_all = np.array([float(row[right_idx]) for row in stimparam], dtype=float)

    left_reward_p = left_all[schedule.astype(int) - 1]
    right_reward_p = right_all[schedule.astype(int) - 1]

    # MATLAB global rpbox trial_events: columns ~ [trial, time, state, chan, next_state]
    global_trial_events = _ensure_2d_numeric(data["exper"]["rpbox"]["param"]["trial_events"]["value"])
    trial_cells = _collect_trial_event_matrices(param["trial_events"]["trial"])

    delay_odor = int(np.squeeze(param["delayodor"]["value"]))

    rlwm_rows: List[List[float]] = []
    valid_trials = np.zeros(counted_trial, dtype=bool)

    kk = 0
    tt2 = 0.0

    for k in range(counted_trial):
        trial_k = trial_cells[k] if k < len(trial_cells) else np.empty((0, 3), dtype=float)

        if k == 0:
            tt1 = 0.0
            if trial_k.size > 0:
                if result[k] in (1.2, 1.3):
                    tt2 = trial_k[-1, 2]
                else:
                    tt2 = trial_k[0, 2]
                kk += 1
            else:
                tt2 = 0.0
        else:
            tt1 = tt2
            if trial_k.size > 0:
                if result[k] in (1.2, 1.3):
                    tt2 = trial_k[-1, 2]
                else:
                    tt2 = trial_k[0, 2]
                kk += 1
            else:
                if result[k] == 2 and k < counted_trial - 1:
                    raise ValueError(f"no trial events in odor_rlwm for trial {k+1}, file: {filename}")
                elif result[k] == 0 and k < counted_trial - 1:
                    next_trial = trial_cells[k + 1] if (k + 1) < len(trial_cells) else np.empty((0, 3))
                    if next_trial.size == 0:
                        raise ValueError(f"no trial events in odor_rlwm for trial {k+2}, file: {filename}")

                    tt3 = next_trial[0, 2]
                    temp_te = global_trial_events[
                        (global_trial_events[:, 1] > tt1) & (global_trial_events[:, 1] < tt3), 1:4
                    ]
                    trans_mask = (temp_te[:, 1] == 512) & (temp_te[:, 2] == 8)
                    trans_inds = np.where(trans_mask)[0]

                    if len(trans_inds) == 1:
                        # MATLAB version ignores current trial here
                        pass
                    elif len(trans_inds) == 2:
                        tt2 = temp_te[trans_inds[1], 0]
                    else:
                        raise ValueError(f"no trial events in odor_rlwm for trial {k+1}, file: {filename}")
                else:
                    raise ValueError(f"no trial events in odor_rlwm for trial {k+1}, file: {filename}")

        current_te = global_trial_events[
            (global_trial_events[:, 1] > tt1) & (global_trial_events[:, 1] <= tt2), 1:4
        ]  # [time, state, chan]

        if delay_odor == 1:
            odor_on_mask = np.isin(current_te[:, 1], [2, 12, 22]) & (current_te[:, 2] == 8)
        else:
            odor_on_mask = np.isin(current_te[:, 1], [1, 11, 21]) & (current_te[:, 2] == 8)

        odor_on_times = current_te[odor_on_mask, 0]

        if len(odor_on_times) > 0:
            if len(odor_on_times) == 1:
                new_trial_odor_on_time = odor_on_times[0]
            elif len(odor_on_times) == 2 and k == 0:
                new_trial_odor_on_time = odor_on_times[1]
            elif len(odor_on_times) <= 4 and k > 0 and result[k - 1] != 0:
                new_trial_odor_on_time = odor_on_times[-1]
            elif len(odor_on_times) == 2 and k > 0 and result[k - 1] == 0:
                new_trial_odor_on_time = odor_on_times[1]
            else:
                new_trial_odor_on_time = odor_on_times[-1]

            valid_trials[k] = True

            iti_te = global_trial_events[
                (global_trial_events[:, 1] > tt1)
                & (global_trial_events[:, 1] < new_trial_odor_on_time)
                & np.isin(global_trial_events[:, 3], [1, 2, 3, 4, 5, 6]),
                1:4,
            ].copy()  # [time, state, chan]

            if iti_te.size > 0:
                last_poke_out = np.where(np.isin(iti_te[:, 2], [4, 6]))[0]
                if len(last_poke_out) > 0:
                    idx = last_poke_out[-1]
                    iti_te[idx, 2] = iti_te[idx, 2] * 10 + iti_te[idx, 2]  # 4->44, 6->66

                for row in iti_te:
                    rlwm_rows.append([row[2], row[0], kk - 0.5])

            odor_id = odor_channel[k] / 100.0
            rlwm_rows.append([7.0 + odor_id, new_trial_odor_on_time, kk])

            tk_te = global_trial_events[
                (global_trial_events[:, 1] > new_trial_odor_on_time)
                & (global_trial_events[:, 1] <= tt2)
                & np.isin(global_trial_events[:, 3], [1, 2, 3, 4, 5, 6]),
                1:4,
            ].copy()  # [time, state, chan]

            tk_extra = []

            tk_te1 = global_trial_events[
                (global_trial_events[:, 1] > new_trial_odor_on_time)
                & (global_trial_events[:, 1] <= tt2)
                & (global_trial_events[:, 2] == 45)
                & (global_trial_events[:, 3] == 8),
                1:4,
            ]
            for row in tk_te1:
                tk_extra.append([row[0], row[1], 9.01])

            tk_te2 = global_trial_events[
                (global_trial_events[:, 1] > new_trial_odor_on_time)
                & (global_trial_events[:, 1] <= tt2)
                & (global_trial_events[:, 2] == 44)
                & (global_trial_events[:, 3] == 8),
                1:4,
            ]
            for row in tk_te2:
                tk_extra.append([row[0], row[1], 9.02])

            tk_te3 = global_trial_events[
                (global_trial_events[:, 1] > new_trial_odor_on_time)
                & (global_trial_events[:, 1] <= tt2)
                & (global_trial_events[:, 2] == 43)
                & (global_trial_events[:, 3] == 8),
                1:4,
            ]
            for row in tk_te3:
                tk_extra.append([row[0], row[1], 9.03])

            tk_all = []
            if tk_te.size > 0:
                tk_all.extend(tk_te.tolist())
            tk_all.extend(tk_extra)

            if len(tk_all) > 0:
                tk_all = np.array(tk_all, dtype=float)
                tk_all = tk_all[np.argsort(tk_all[:, 0])]
                for row in tk_all:
                    rlwm_rows.append([row[2], row[0], kk])

            rlwm_rows.append([80.0 + result[k], tt2, kk])

        elif tt1 == tt2:
            # skip current trial
            pass
        else:
            raise ValueError(f"no odor on time for trial {k+1} in file: {filename}")

    if len(rlwm_rows) > 0:
        rlwm_event_times = np.array(rlwm_rows, dtype=float).T
    else:
        rlwm_event_times = np.empty((3, 0), dtype=float)

    portside = portside.copy()
    portside[(left_reward_p == -1) & (right_reward_p == -1)] = -1

    return RLWMParsed(
        rlwm_event_times=rlwm_event_times,
        odor_name=odor_name[valid_trials],
        odor_dur=odor_dur[valid_trials],
        schedule=schedule[valid_trials],
        portside=portside[valid_trials],
        result=result[valid_trials],
        protocol_name=protocol_name,
    )


def extract_behavior_df_py(filename: str) -> pd.DataFrame:
    """
    Python port of extract_behavior_df.m.
    """
    parsed = get_rlwm_event_times_py(filename)
    dmat = parsed.rlwm_event_times.T

    if dmat.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "trial",
                "outcome",
                "center_in",
                "center_out",
                "side_in",
                "last_side_out",
                "actions",
                "reward",
                "trial_types",
                "odors",
                "port_side",
                "schedule",
                "odor_name",
                "odor_dur",
            ]
        )

    outcome_inds = np.where(dmat[:, 0] > 80)[0]
    odor_inds = np.where(np.floor(dmat[:, 0]) == 7)[0]

    out = {}
    out["trial"] = np.arange(1, len(outcome_inds) + 1, dtype=float)

    # Keep compatibility with current MATLAB helper behavior:
    # this is the outcome event time, not the categorical outcome code.
    out["outcome"] = dmat[outcome_inds, 1]

    out["center_in"] = _backward_times(
        dmat, outcome_inds, lambda region: region[:, 0] == 1
    )
    out["center_out"] = _backward_times(
        dmat, outcome_inds, lambda region: region[:, 0] == 2
    )
    out["side_in"] = _backward_times(
        dmat, outcome_inds, lambda region: (region[:, 0] == 3) | (region[:, 0] == 5)
    )

    # Filter out impossible side_in values:
    # valid side choice should occur after center_out within the same trial.
    invalid_side_in = (
        ~np.isnan(out["side_in"])
        & ~np.isnan(out["center_out"])
        & (out["side_in"] < out["center_out"])
    )
    out["side_in"][invalid_side_in] = np.nan

    so_times = np.full(len(outcome_inds), np.nan, dtype=float)
    for i in range(len(outcome_inds)):
        start_ind = outcome_inds[i]
        end_ind = len(dmat) if i == len(outcome_inds) - 1 else odor_inds[i + 1]
        region = dmat[start_ind:end_ind, :]
        so_time = region[(region[:, 0] == 44) | (region[:, 0] == 66), 1]
        if len(so_time) > 0:
            so_times[i] = so_time[-1]
    out["last_side_out"] = so_times

    trial_sel = np.isin(dmat[:, 1], out["side_in"]) & (dmat[:, 0] < 80)

    choice_trials_raw = dmat[trial_sel, 2]
    choice_event_ids = dmat[trial_sel, 0]

    # Only keep integer-numbered trial assignments, not half-trial ITI labels like 2.5
    valid_choice = np.isclose(choice_trials_raw, np.round(choice_trials_raw))

    choice_trials = np.round(choice_trials_raw[valid_choice]).astype(np.uint16)
    choice_event_ids = choice_event_ids[valid_choice]

    actions = np.full(len(outcome_inds), np.nan, dtype=float)
    if len(choice_trials) > 0:
        actions[choice_trials - 1] = (choice_event_ids - 3) / 2

    out["actions"] = actions

    waters = np.full(len(outcome_inds), np.nan, dtype=float)
    water_sel = np.floor(dmat[:, 0]) == 9
    water_given = dmat[water_sel, 2].astype(np.uint16)
    if len(water_given) > 0:
        waters[water_given - 1] = np.mod(dmat[water_sel, 0], 1) * 100
    out["reward"] = waters

    out["trial_types"] = np.mod(dmat[np.floor(dmat[:, 0]) > 80, 0], 1) / 10
    out["odors"] = np.mod(dmat[np.floor(dmat[:, 0]) == 7, 0], 1) * 100

    out["port_side"] = parsed.portside
    out["schedule"] = parsed.schedule
    out["odor_name"] = parsed.odor_name
    out["odor_dur"] = parsed.odor_dur

    return pd.DataFrame(out)