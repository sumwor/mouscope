# utility functions for behaviral analysis
import csv
import gspread
import numpy as np
import pandas as pd

import os
import shutil
from bisect import bisect_left, bisect_right
import imageio
from skimage import color

import matplotlib
matplotlib.use('QtAgg') 
from datetime import datetime, timedelta
import time
from gspread.exceptions import APIError
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()
import matplotlib.pyplot as plt
# Deeplabcut related, and MotionSequence related functions
from scipy.signal import butter, filtfilt
from pygam import LinearGAM, s, f
from scipy.stats import chi2

def load_DLC(filepath):

    # load DLC results
    data = {}
    nFrames = 0

    if isinstance(filepath, str):
        with open(filepath) as csv_file:
            #print("Loading data from: " + filePath)
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # scorer
                    data[row[0]] = row[1]
                    line_count += 1
                elif line_count == 1:  # body parts
                    bodyPartList = []
                    for bb in range(len(row) - 1):
                        if row[bb + 1] not in bodyPartList:
                            bodyPartList.append(row[bb + 1])
                    data[row[0]] = bodyPartList
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 2:  # coords
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 3:  # actual coords
                    # print({", ".join(row)})
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        if np.mod(ii, 3) == 0:
                            data[body] = {}
                        data[body][tempList[np.mod(ii, 3)]] = [float(row[ii + 1])]
                    #self.t.append(0)
                    line_count += 1
                    nFrames += 1

                else:
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                    #self.t.append(self.nFrames*(1/self.fps))
                    line_count += 1
                    nFrames += 1

            print(f'Processed {line_count} lines.')
    
    return data

def moving_average(x, window=5):
    return np.convolve(x, np.ones(window)/window, mode='same')


def gam_learning_analysis(
    perf_df,
    subject_col='subject',
    genotype_col='genotype',
    trial_col='trial',
    value_col='time_on_rod',
    exclude_col='FallByTurning',
    exclude_value=1,
    n_splines=5,
    return_model_data=False,
):
    """
    GAM analysis for learning curves.

    Tests:
    1. genotype effect
    2. learning effect (trial)
    3. genotype × learning interaction

    Parameters
    ----------
    perf_df : pd.DataFrame
        Input dataframe.

    return_model_data : bool
        If True, also return the cleaned dataframe used for fitting.

    Returns
    -------
    stats_df : pd.DataFrame
        Statistical test summary.

    models : dict
        Fitted GAM models.

    model_data : pd.DataFrame
        Cleaned model input, including encoded subject ids.
    """

    # ------------------------------------------------------------
    # Copy dataframe
    # ------------------------------------------------------------
    df = perf_df.copy()

    # ------------------------------------------------------------
    # Treat FallByTurning == 1 as invalid trials
    # ------------------------------------------------------------
    exclude_numeric = pd.to_numeric(df[exclude_col], errors='coerce') == exclude_value
    exclude_text = df[exclude_col].astype(str).str.lower().isin(['true', '1', 'yes'])
    df.loc[exclude_numeric | exclude_text, value_col] = np.nan

    # ------------------------------------------------------------
    # Remove missing data
    # ------------------------------------------------------------
    df = df.dropna(
        subset=[
            subject_col,
            genotype_col,
            trial_col,
            value_col
        ]
    ).copy()

    # ------------------------------------------------------------
    # Convert trial to numeric
    # ------------------------------------------------------------
    if pd.api.types.is_categorical_dtype(df[trial_col]):
        df['trial_num'] = df[trial_col].cat.codes.astype(float) + 1
    else:
        df['trial_num'] = pd.to_numeric(df[trial_col]).astype(float)

    # ------------------------------------------------------------
    # Encode genotype
    # ------------------------------------------------------------
    df[genotype_col] = df[genotype_col].astype('category')

    genotype_levels = list(df[genotype_col].cat.categories)

    if len(genotype_levels) != 2:
        raise ValueError(
            f'Expected exactly 2 genotypes, got: {genotype_levels}'
        )

    df['genotype_code'] = df[genotype_col].cat.codes.astype(int)

    # ------------------------------------------------------------
    # Encode subject for repeated measurements from the same animal.
    # This factor is included in every model below.
    # ------------------------------------------------------------
    df[subject_col] = df[subject_col].astype('category')
    df['subject_code'] = df[subject_col].cat.codes.astype(int)

    # ------------------------------------------------------------
    # Interaction column
    # reference genotype = first category
    # ------------------------------------------------------------
    interaction_genotype = genotype_levels[1]

    df['interaction_by'] = (
        df[genotype_col] == interaction_genotype
    ).astype(float)

    # ------------------------------------------------------------
    # Design matrix
    # column 0 = trial_num
    # column 1 = genotype_code
    # column 2 = interaction_by
    # column 3 = subject_code
    # ------------------------------------------------------------
    X = df[
        ['trial_num', 'genotype_code', 'interaction_by', 'subject_code']
    ].to_numpy(dtype=float)

    y = df[value_col].to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    # no genotype
    no_genotype_model = LinearGAM(
        s(0, n_splines=n_splines) +
        f(3)
    ).fit(X, y)

    # genotype main effect
    genotype_model = LinearGAM(
        s(0, n_splines=n_splines) +
        f(1) +
        f(3)
    ).fit(X, y)

    # no learning
    no_learning_model = LinearGAM(
        f(1) +
        f(3)
    ).fit(X, y)

    # full interaction model
    full_model = LinearGAM(
        s(0, n_splines=n_splines) +
        f(1) +
        s(0, by=2, n_splines=n_splines) +
        f(3)
    ).fit(X, y)

    # ------------------------------------------------------------
    # Likelihood ratio test
    # ------------------------------------------------------------
    def lr_test(reduced_model, full_model, term_name):

        lr_stat = 2 * (
            full_model.statistics_['loglikelihood'] -
            reduced_model.statistics_['loglikelihood']
        )

        df_diff = (
            full_model.statistics_['edof'] -
            reduced_model.statistics_['edof']
        )

        if df_diff <= 0:
            p_value = np.nan
        else:
            p_value = chi2.sf(lr_stat, df_diff)

        return {
            'term': term_name,
            'lr_stat': lr_stat,
            'df': df_diff,
            'p_value': p_value,
        }

    # ------------------------------------------------------------
    # Statistical summary
    # ------------------------------------------------------------
    stats_df = pd.DataFrame([

        # genotype effect
        lr_test(
            no_genotype_model,
            genotype_model,
            'genotype'
        ),

        # learning effect
        lr_test(
            no_learning_model,
            genotype_model,
            'learning'
        ),

        # interaction effect
        lr_test(
            genotype_model,
            full_model,
            'genotype:learning'
        )

    ])

    # ------------------------------------------------------------
    # Add metadata
    # ------------------------------------------------------------
    stats_df['n_subjects'] = df[subject_col].nunique()
    stats_df['n_observations'] = len(df)
    stats_df['n_splines'] = n_splines
    stats_df['reference_genotype'] = genotype_levels[0]
    stats_df['interaction_genotype'] = interaction_genotype
    stats_df['repeated_measure_term'] = subject_col

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    models = {
        'no_genotype_model': no_genotype_model,
        'genotype_model': genotype_model,
        'no_learning_model': no_learning_model,
        'full_model': full_model,
    }

    model_data = df[
        [
            subject_col,
            genotype_col,
            trial_col,
            value_col,
            'trial_num',
            'genotype_code',
            'interaction_by',
            'subject_code',
        ]
    ].copy()

    if return_model_data:
        return stats_df, models, model_data

    return stats_df, models

def run_learning_gam(perf_df, summary_path):

    stats_df = pd.DataFrame()

    perf_df = perf_df.copy()


    if 'trial' not in perf_df.columns:
        if 'block' in perf_df.columns:
            perf_df.rename(columns={'block': 'trial'}, inplace=True)
        elif 'trial/block' in perf_df.columns:
            perf_df.rename(columns={'trial/block': 'trial'}, inplace=True)

    if 'FallByTurning' in perf_df.columns:
        exclude_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
        exclude_text = perf_df['FallByTurning'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
        perf_df.loc[exclude_numeric | exclude_text, 'time_on_rod'] = np.nan

    # ------------------------------------------------------------
    # Clean data
    # ------------------------------------------------------------
    clean_df = perf_df.dropna(
        subset=['subject', 'genotype', 'trial', 'performance']
    ).copy()

    if clean_df['genotype'].nunique() > 1 and clean_df['trial'].nunique() > 1:

        gam_df = clean_df.copy()

        # ------------------------------------------------------------
        # Encode categorical variables
        # ------------------------------------------------------------
        gam_df['subject'] = gam_df['subject'].astype('category')
        gam_df['genotype'] = gam_df['genotype'].astype('category')

        gam_df['subject_code'] = gam_df['subject'].cat.codes.astype(int)
        gam_df['genotype_code'] = gam_df['genotype'].cat.codes.astype(int)

        genotype_levels = list(gam_df['genotype'].cat.categories)

        # stable reference genotype
        reference_genotype = (
            'WT' if 'WT' in genotype_levels else genotype_levels[0]
        )

        reordered = [
            reference_genotype
        ] + [
            g for g in genotype_levels if g != reference_genotype
        ]

        gam_df['genotype'] = gam_df['genotype'].cat.reorder_categories(
            reordered
        )

        genotype_levels = reordered

        # ------------------------------------------------------------
        # Trial handling
        # ------------------------------------------------------------
        try:
            gam_df['trial_num'] = pd.to_numeric(gam_df['trial']).astype(float)
        except Exception:
            gam_df['trial'] = gam_df['trial'].astype('category')
            gam_df['trial_num'] = gam_df['trial'].cat.codes.astype(float) + 1

        # ------------------------------------------------------------
        # Interaction terms
        # ------------------------------------------------------------
        interaction_levels = genotype_levels[1:]

        x_columns = ['trial_num', 'genotype_code']

        for g in interaction_levels:
            col = f'by_{g}'
            gam_df[col] = (gam_df['genotype'] == g).astype(float)
            x_columns.append(col)

        subject_col_idx = len(x_columns)
        x_columns.append('subject_code')

        X = gam_df[x_columns].to_numpy(dtype=float)
        y = gam_df['performance'].to_numpy(dtype=float)

        # ------------------------------------------------------------
        # Spline selection
        # ------------------------------------------------------------
        n_trials = gam_df['trial_num'].nunique()
        n_splines = int(min(max(n_trials, 4), 6))

        # ------------------------------------------------------------
        # MODELS
        # ------------------------------------------------------------

        no_genotype_model = LinearGAM(
            s(0, n_splines=n_splines) +
            f(subject_col_idx)
        ).gridsearch(X, y, progress=False)

        genotype_model = LinearGAM(
            s(0, n_splines=n_splines) +
            f(1) +
            f(subject_col_idx)
        ).gridsearch(X, y, progress=False)

        no_learning_model = LinearGAM(
            f(1) +
            f(subject_col_idx)
        ).gridsearch(X, y, progress=False)

        full_terms = (
            s(0, n_splines=n_splines) +
            f(1)
        )

        for col_idx in range(2, subject_col_idx):
            full_terms += s(0, by=col_idx, n_splines=n_splines)

        full_terms += f(subject_col_idx)

        full_model = LinearGAM(full_terms).gridsearch(
            X, y, progress=False
        )

        # ------------------------------------------------------------
        # Likelihood ratio test
        # ------------------------------------------------------------
        def lr_test(reduced, full, term):
            lr_stat = 2 * (
                full.statistics_['loglikelihood'] -
                reduced.statistics_['loglikelihood']
            )

            df_diff = (
                full.statistics_['edof'] -
                reduced.statistics_['edof']
            )

            p_value = (
                chi2.sf(lr_stat, df_diff)
                if df_diff > 0 else np.nan
            )

            return {
                'term': term,
                'lr_stat': lr_stat,
                'df': df_diff,
                'p_value': p_value,
            }

        stats_df = pd.DataFrame([
            lr_test(no_genotype_model, genotype_model, 'genotype'),
            lr_test(no_learning_model, genotype_model, 'learning'),
            lr_test(genotype_model, full_model, 'genotype:learning')
        ])

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------
        stats_df['n_observations'] = int(gam_df.shape[0])
        stats_df['n_subjects'] = int(gam_df['subject'].nunique())
        stats_df['n_trials'] = int(n_trials)
        stats_df['n_splines'] = int(n_splines)
        stats_df['genotype_reference'] = reference_genotype
        stats_df['genotype_levels'] = ','.join(genotype_levels)
        stats_df['repeated_measure_term'] = 'subject_fixed_effect'
        stats_df['x_columns'] = ','.join(x_columns)

        if summary_path is not None:
            os.makedirs(summary_path, exist_ok=True)
            stats_df.to_csv(
                os.path.join(
                    summary_path,
                    'Rotarod performance gam.csv'
                ),
                index=False
            )

        models = {
            'no_genotype_model': no_genotype_model,
            'genotype_model': genotype_model,
            'no_learning_model': no_learning_model,
            'full_model': full_model,
        }

        return stats_df, models, gam_df

    else:
        stats_df = pd.DataFrame({
            'error': ['Insufficient genotype or trial variation']
        })

        return stats_df, None, None
    

def run_learning_gamm(perf_df, summary_path):

    import statsmodels.api as sm
    from patsy import dmatrix

    stats_df = pd.DataFrame()

    perf_df = perf_df.copy()

    if 'time_on_rod' not in perf_df.columns and 'performance' in perf_df.columns:
        perf_df.rename(columns={'performance': 'time_on_rod'}, inplace=True)

    if 'trial' not in perf_df.columns:
        if 'block' in perf_df.columns:
            perf_df.rename(columns={'block': 'trial'}, inplace=True)
        elif 'trial/block' in perf_df.columns:
            perf_df.rename(columns={'trial/block': 'trial'}, inplace=True)

    if 'FallByTurning' in perf_df.columns:
        exclude_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
        exclude_text = perf_df['FallByTurning'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
        perf_df.loc[exclude_numeric | exclude_text, 'time_on_rod'] = np.nan

    clean_df = perf_df.dropna(
        subset=['subject', 'genotype', 'trial', 'time_on_rod']
    ).copy()

    if clean_df['genotype'].nunique() <= 1 or clean_df['trial'].nunique() <= 1:
        stats_df = pd.DataFrame({
            'error': ['Insufficient genotype or trial variation']
        })
        return stats_df, None, None

    gamm_df = clean_df.copy()
    gamm_df['subject'] = gamm_df['subject'].astype('category')
    gamm_df['genotype'] = gamm_df['genotype'].astype('category')

    genotype_levels = list(gamm_df['genotype'].cat.categories)
    reference_genotype = 'WT' if 'WT' in genotype_levels else genotype_levels[0]
    genotype_levels = [reference_genotype] + [
        g for g in genotype_levels if g != reference_genotype
    ]
    gamm_df['genotype'] = gamm_df['genotype'].cat.reorder_categories(
        genotype_levels
    )

    try:
        gamm_df['trial_num'] = pd.to_numeric(gamm_df['trial']).astype(float)
    except Exception:
        gamm_df['trial'] = gamm_df['trial'].astype('category')
        gamm_df['trial_num'] = gamm_df['trial'].cat.codes.astype(float) + 1

    n_trials = gamm_df['trial_num'].nunique()
    n_splines = int(min(max(n_trials, 4), 6))

    spline_df = dmatrix(
        f'bs(trial_num, df={n_splines}, degree=3, include_intercept=False) - 1',
        gamm_df,
        return_type='dataframe'
    )
    spline_df.columns = [f'spline_{i}' for i in range(spline_df.shape[1])]
    spline_df.index = gamm_df.index

    genotype_df = pd.get_dummies(
        gamm_df['genotype'],
        drop_first=True,
        dtype=float
    )
    genotype_df.columns = [f'genotype_{col}' for col in genotype_df.columns]
    genotype_df.index = gamm_df.index

    interaction_parts = []
    for genotype_col in genotype_df.columns:
        interaction_parts.append(
            spline_df.multiply(genotype_df[genotype_col], axis=0).add_prefix(
                f'{genotype_col}:'
            )
        )

    def design_matrix(parts):
        exog = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=gamm_df.index)
        exog = sm.add_constant(exog, has_constant='add')
        return exog.astype(float)

    no_genotype_X = design_matrix([spline_df])
    genotype_X = design_matrix([spline_df, genotype_df])
    no_learning_X = design_matrix([genotype_df])
    full_X = design_matrix([spline_df, genotype_df] + interaction_parts)

    y = gamm_df['time_on_rod'].astype(float)
    groups = gamm_df['subject']

    def fit_model(exog):
        model = sm.MixedLM(y, exog, groups=groups)
        try:
            return model.fit(reml=False, method='lbfgs', disp=False)
        except Exception:
            return model.fit(reml=False, method='powell', disp=False)

    no_genotype_model = fit_model(no_genotype_X)
    genotype_model = fit_model(genotype_X)
    no_learning_model = fit_model(no_learning_X)
    full_model = fit_model(full_X)

    def lr_test(reduced, full, term):
        lr_stat = 2 * (full.llf - reduced.llf)
        df_diff = len(full.fe_params) - len(reduced.fe_params)
        p_value = chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
        return {
            'term': term,
            'lr_stat': lr_stat,
            'df': df_diff,
            'p_value': p_value,
        }

    stats_df = pd.DataFrame([
        lr_test(no_genotype_model, genotype_model, 'genotype'),
        lr_test(no_learning_model, genotype_model, 'learning'),
        lr_test(genotype_model, full_model, 'genotype:learning')
    ])

    stats_df['n_observations'] = int(gamm_df.shape[0])
    stats_df['n_subjects'] = int(gamm_df['subject'].nunique())
    stats_df['n_trials'] = int(n_trials)
    stats_df['n_splines'] = int(n_splines)
    stats_df['genotype_reference'] = reference_genotype
    stats_df['genotype_levels'] = ','.join(genotype_levels)
    stats_df['repeated_measure_term'] = 'subject_random_intercept'
    stats_df['x_columns'] = ','.join(full_X.columns)

    if summary_path is not None:
        os.makedirs(summary_path, exist_ok=True)
        stats_df.to_csv(
            os.path.join(summary_path, 'Learning performance gamm.csv'),
            index=False
        )

    models = {
        'no_genotype_model': no_genotype_model,
        'genotype_model': genotype_model,
        'no_learning_model': no_learning_model,
        'full_model': full_model,
    }

    model_data = gamm_df[
        ['subject', 'genotype', 'trial', 'trial_num', 'time_on_rod']
    ].copy()

    return stats_df, models, model_data


def run_learning_FDA(
    perf_df,
    save_name,
    summary_path,
    n_permutations=1000,
    random_state=0,
    min_points=2,
):

    stats_df = pd.DataFrame()
    perf_df = perf_df.copy()


    if 'trial' not in perf_df.columns:
        if 'block' in perf_df.columns:
            perf_df.rename(columns={'block': 'trial'}, inplace=True)
        elif 'trial/block' in perf_df.columns:
            perf_df.rename(columns={'trial/block': 'trial'}, inplace=True)

    if 'FallByTurning' in perf_df.columns:
        exclude_numeric = pd.to_numeric(perf_df['FallByTurning'], errors='coerce') == 1
        exclude_text = perf_df['FallByTurning'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
        perf_df.loc[exclude_numeric | exclude_text, 'performance'] = np.nan

    #perf_df['performance'] = pd.to_numeric(perf_df['performance'], errors='coerce')
    clean_df = perf_df.dropna(
        subset=['subject', 'genotype', 'trial', 'performance']
    ).copy()

    if clean_df['genotype'].nunique() <= 1 or clean_df['trial'].nunique() <= 1:
        stats_df = pd.DataFrame({
            'error': ['Insufficient genotype or trial variation']
        })
        return stats_df, None, None

    clean_df['subject'] = clean_df['subject'].astype('category')
    clean_df['genotype'] = clean_df['genotype'].astype('category')

    genotype_levels = list(clean_df['genotype'].cat.categories)
    reference_genotype = 'WT' if 'WT' in genotype_levels else genotype_levels[0]
    genotype_levels = [reference_genotype] + [
        g for g in genotype_levels if g != reference_genotype
    ]
    clean_df['genotype'] = clean_df['genotype'].cat.reorder_categories(
        genotype_levels
    )

    trial_numeric = pd.to_numeric(clean_df['trial'], errors='coerce')
    if trial_numeric.notna().all():
        clean_df['trial_num'] = trial_numeric.astype(float)
    else:
        clean_df['trial'] = clean_df['trial'].astype('category')
        clean_df['trial_num'] = clean_df['trial'].cat.codes.astype(float) + 1

    trial_grid = np.sort(clean_df['trial_num'].unique().astype(float))
    grouped = clean_df.groupby(
        ['subject', 'genotype', 'trial_num'],
        observed=True
    )['performance'].mean().reset_index()

    subject_ids = []
    subject_genotypes = []
    curves = []
    for subject, subject_df in grouped.groupby('subject', observed=True):
        trial_values = subject_df['trial_num'].to_numpy(dtype=float)
        if np.unique(trial_values).size < min_points:
            continue

        subject_df = subject_df.sort_values('trial_num')
        subject_ids.append(subject)
        subject_genotypes.append(subject_df['genotype'].iloc[0])
        curves.append(
            np.interp(
                trial_grid,
                subject_df['trial_num'].to_numpy(dtype=float),
                subject_df['performance'].to_numpy(dtype=float)
            )
        )

    if len(curves) < 2 or len(np.unique(subject_genotypes)) < 2:
        stats_df = pd.DataFrame({
            'error': ['Insufficient subject curves or genotype variation']
        })
        return stats_df, None, None

    curves = np.asarray(curves, dtype=float)
    subject_genotypes = np.asarray(subject_genotypes)
    rng = np.random.default_rng(random_state)

    def curve_integral(values):
        return float(np.trapz(values, trial_grid))

    def genotype_stat(curve_matrix, labels):
        grand_mean = curve_matrix.mean(axis=0)
        stat = 0.0
        for genotype in np.unique(labels):
            group_curves = curve_matrix[labels == genotype]
            diff = group_curves.mean(axis=0) - grand_mean
            stat += group_curves.shape[0] * curve_integral(diff ** 2)
        return float(stat)

    def learning_stat(curve_matrix):
        mean_curve = curve_matrix.mean(axis=0)
        centered_mean = mean_curve - mean_curve.mean()
        return curve_integral(centered_mean ** 2)

    def interaction_stat(curve_matrix, labels):
        grand_centered = curve_matrix.mean(axis=0)
        grand_centered = grand_centered - grand_centered.mean()
        stat = 0.0
        for genotype in np.unique(labels):
            group_curves = curve_matrix[labels == genotype]
            group_centered = group_curves.mean(axis=0)
            group_centered = group_centered - group_centered.mean()
            diff = group_centered - grand_centered
            stat += group_curves.shape[0] * curve_integral(diff ** 2)
        return float(stat)

    observed = {
        'genotype': genotype_stat(curves, subject_genotypes),
        'learning': learning_stat(curves),
        'genotype:learning': interaction_stat(curves, subject_genotypes),
    }

    null_stats = {term: np.empty(n_permutations) for term in observed}
    for perm_idx in range(n_permutations):
        permuted_labels = rng.permutation(subject_genotypes)
        null_stats['genotype'][perm_idx] = genotype_stat(curves, permuted_labels)
        null_stats['genotype:learning'][perm_idx] = interaction_stat(
            curves,
            permuted_labels
        )

        permuted_curves = curves.copy()
        for row_idx in range(permuted_curves.shape[0]):
            permuted_curves[row_idx] = rng.permutation(permuted_curves[row_idx])
        null_stats['learning'][perm_idx] = learning_stat(permuted_curves)

    rows = []
    for term, stat in observed.items():
        if n_permutations > 0:
            p_value = (
                np.sum(null_stats[term] >= stat) + 1
            ) / (n_permutations + 1)
        else:
            p_value = np.nan

        rows.append({
            'term': term,
            'statistic': stat,
            'p_value': p_value,
        })

    stats_df = pd.DataFrame(rows)
    stats_df['n_observations'] = int(clean_df.shape[0])
    stats_df['n_subjects'] = int(len(subject_ids))
    stats_df['n_trials'] = int(len(trial_grid))
    stats_df['n_permutations'] = int(n_permutations)
    stats_df['genotype_reference'] = reference_genotype
    stats_df['genotype_levels'] = ','.join(genotype_levels)
    stats_df['repeated_measure_term'] = 'subject_level_function'
    stats_df['method'] = 'functional_permutation_test'

    if summary_path is not None:
        os.makedirs(os.path.join(summary_path, 'Results'), exist_ok=True)
        stats_df.to_csv(
            os.path.join(summary_path,'Results', f'{save_name}  FDA.csv'),
            index=False
        )

    model_data = pd.DataFrame(
        curves,
        index=pd.Index(subject_ids, name='subject'),
        columns=[f'trial_{trial:g}' for trial in trial_grid]
    ).reset_index()
    model_data['genotype'] = subject_genotypes

    models = {
        'trial_grid': trial_grid,
        'subject_curves': curves,
        'subject_ids': subject_ids,
        'subject_genotypes': subject_genotypes,
        'observed_statistics': observed,
        'null_statistics': null_stats,
    }

    return stats_df, models, model_data


def plot_learning_curve(
    perf_df,
    summary_path=None,
    save_name='Learning curve',
    value_col='performance',
    trial_col=None,
    ylabel='Performance',
    title='Learning Curve',
    ax=None,
    show_raw=True,
):
   

    plot_df = perf_df.copy()
    plot_df.rename(
        columns={
            'Animal': 'subject',
            'Genotype': 'genotype',
            trial_col: 'trial',
            value_col: 'performance',
        },
        inplace=True
    )


    if 'FallByTurning' in plot_df.columns:
        if plot_df['FallByTurning'].dtype == bool:
            exclude_mask = plot_df['FallByTurning'].fillna(False)
        else:
            exclude_numeric = pd.to_numeric(plot_df['FallByTurning'], errors='coerce') == 1
            exclude_text = plot_df['FallByTurning'].astype(str).str.lower().isin(
                ['true', '1', 'yes']
            )
            exclude_mask = exclude_numeric | exclude_text
        plot_df.loc[exclude_mask, 'performance'] = np.nan

    #plot_df['performance'] = pd.to_numeric(plot_df['performance'], errors='coerce')
    clean_df = plot_df.dropna(
        subset=['subject', 'genotype', 'trial', 'performance']
    ).copy()

    trial_order = pd.unique(clean_df['trial'].dropna())
    trial_numeric = pd.to_numeric(pd.Series(trial_order), errors='coerce')
    if trial_numeric.notna().all():
        trial_order = trial_order[np.argsort(trial_numeric.to_numpy())]

    clean_df['trial'] = pd.Categorical(
        clean_df['trial'],
        categories=trial_order,
        ordered=True
    )

    genotype_order = [
        g for g in ['WT', 'HET', 'KO'] if g in set(clean_df['genotype'].dropna())
    ]
    genotype_order += [
        g for g in pd.unique(clean_df['genotype'].dropna()) if g not in genotype_order
    ]
    clean_df['genotype'] = pd.Categorical(
        clean_df['genotype'],
        categories=genotype_order,
        ordered=True
    )


    stats_df, _, _ = run_learning_FDA(clean_df, save_name=save_name,summary_path=summary_path)

    summary_df = clean_df.groupby(
        ['genotype', 'trial'],
        observed=True
    )['performance'].agg(['mean', 'std', 'count']).reset_index()
    summary_df['sem'] = summary_df['std'] / np.sqrt(summary_df['count'])
    genotype_counts = clean_df.groupby(
        'genotype',
        observed=True
    )['subject'].nunique()
    trial_codes = {trial: idx for idx, trial in enumerate(trial_order)}

    # find the max performance to determine plot limits
    max_performance = clean_df['performance'].max()
    if max_performance <= 1:
        ymax = 1
    elif max_performance > 50 and max_performance <= 300:
        ymax = 300


    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    colors = {'WT': 'black', 'HET': 'red', 'KO': 'red'}
    for genotype in genotype_order:
        genotype_data = summary_df[summary_df['genotype'] == genotype]
        if genotype_data.empty:
            continue

        x = genotype_data['trial'].map(trial_codes).astype(float).to_numpy()
        y = genotype_data['mean'].to_numpy(dtype=float)
        sem = genotype_data['sem'].to_numpy(dtype=float)
        color = colors.get(genotype, None)
        label = f'{genotype} (n={genotype_counts.get(genotype, 0)})'
        ax.errorbar(
            x, y, yerr=sem, marker='o', linewidth=2, capsize=3,
            color=color, label=label
        )

        if show_raw:
            raw = clean_df[clean_df['genotype'] == genotype]
            for subject, subject_df in raw.groupby('subject', observed=True):
                subject_x = subject_df['trial'].map(trial_codes).astype(float).to_numpy()
                subject_y = subject_df['performance'].to_numpy(dtype=float)
                if subject_x.size == 0:
                    continue
                sort_idx = np.argsort(subject_x)
                ax.plot(
                    subject_x[sort_idx],
                    subject_y[sort_idx],
                    color=color,
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.35
                )

    if not stats_df.empty and 'p_value' in stats_df.columns:
        p_values = stats_df.set_index('term')['p_value']
        stats_text = (
            f"FDA p genotype = {p_values.get('genotype', np.nan):.3g}\n"
            f"FDA p learning = {p_values.get('learning', np.nan):.3g}\n"
            f"FDA p interaction = {p_values.get('genotype:learning', np.nan):.3g}"
        )
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            va='top',
            ha='right',
            fontsize=9
        )

    # save stats to csv
    # if summary_path is not None:
    #     os.makedirs(os.path.join(summary_path, 'Results'), exist_ok=True)
    #     stats_df.to_csv(
    #         os.path.join(summary_path, 'Results', f'{save_name}_FDA.csv'),
    #         index=False
    #     )

    # plot 0.5 and 0.7 line in the plot
    ax.axhline(y=0.5, color=[0.7, 0.7, 0,.7], linestyle='--')
    ax.axhline(y=0.7, color=[0.7, 0.7, 0,.7], linestyle='--')
    ax.set_xticks(np.arange(len(trial_order)))
    ax.set_xticklabels([str(t) for t in trial_order])
    ax.set_xlabel('Trial / Block')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if summary_path is not None:
        os.makedirs(os.path.join(summary_path, 'BehPlots'), exist_ok=True)
        fig.savefig(os.path.join(summary_path, 'BehPlots', f'{save_name}.png'), dpi=300)
        fig.savefig(os.path.join(summary_path, 'BehPlots', f'{save_name}.svg'), format='svg')

    plt.close(fig)
    #return fig, ax, stats_df, clean_df

def plot_session(resultdf, protocol, save_path=None, label=None):
    # for each odor behavior session, plot the performance
    # first subplot
    # plot the performance of each session and running average reward rate

    # input:
    # resultdf: result data frame read from the behavior csv file
    # protocol: the training protocol (AB, AB-CD.. etc)
    # save_path: optional directory where the figure files will be saved
    # tlabel: optional label for the figure title and filenames

    # disable interactive mode
    
    plt.ioff()

    save_name = f'session-beh_{label}'
    png_path = os.path.join(save_path, f'{save_name}.png')
    if not os.path.exists(png_path):
   
        os.makedirs(save_path, exist_ok=True)

        n_plot = int(len(resultdf))

        schedule = pd.to_numeric(resultdf['schedule'], errors='coerce').astype(int).to_numpy()
        reward = pd.to_numeric(resultdf['reward'], errors='coerce').fillna(0).to_numpy()
        actions = pd.to_numeric(resultdf['actions'], errors='coerce').values

        value2Plot = ((-1) ** schedule) * np.ceil(schedule / 2)
        max_val = int(np.nanmax(np.abs(value2Plot))) if n_plot > 0 else 0
        x = np.arange(1, n_plot + 1)

        def reward_rate(code, start_idx, end_idx):
            if start_idx > end_idx or start_idx < 0 or end_idx < 0:
                return np.nan
            end_idx = min(end_idx, n_plot - 1)
            segment = schedule[start_idx:end_idx + 1]
            segment_reward = reward[start_idx:end_idx + 1]
            mask = segment == code
            denom = np.sum(mask)
            if denom == 0:
                return np.nan
            return np.sum(np.logical_and(mask, segment_reward > 0)) / denom

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        ax = axs[0]
        ax.bar(x, value2Plot * (value2Plot < 0), width=1.0, color='red', edgecolor='none', align='center')
        ax.bar(x, value2Plot * (value2Plot > 0), width=1.0, color='blue', edgecolor='none', align='center')

        if n_plot > 0:
            reward_marker_y = max_val + 0.25
            for ii in range(n_plot):
                if reward[ii] > 0:
                    y = reward_marker_y if actions[ii] == 1 else -reward_marker_y
                    ax.scatter(x[ii], y, marker='.', color='red')

        switch_trial = [np.nan, np.nan]
        if max_val >= 2:
            cdswitch = np.where(np.abs(value2Plot) == 2)[0]
            if cdswitch.size > 0:
                switch_trial[0] = cdswitch[0] + 1
            if max_val == 3:
                revswitch = np.where(np.abs(value2Plot) == 3)[0]
                if revswitch.size > 0:
                    switch_trial[1] = revswitch[0] + 1

        leftRewardRate = [np.nan, np.nan, np.nan]
        rightRewardRate = [np.nan, np.nan, np.nan]

        def to_index(value):
            return int(value) - 1 if not np.isnan(value) else None

        switch_idx_1 = to_index(switch_trial[0])
        switch_idx_2 = to_index(switch_trial[1])

        if switch_idx_1 is None and switch_idx_2 is None:
            leftRewardRate[0] = reward_rate(1, 0, n_plot - 1)
            rightRewardRate[0] = reward_rate(2, 0, n_plot - 1)
        else:
            if switch_idx_2 is None:
                leftRewardRate[0] = reward_rate(1, 0, switch_idx_1)
                rightRewardRate[0] = reward_rate(2, 0, switch_idx_1)
                leftRewardRate[1] = reward_rate(3, switch_idx_1, n_plot - 1)
                rightRewardRate[1] = reward_rate(4, switch_idx_1, n_plot - 1)
            elif switch_idx_1 is None:
                leftRewardRate[0] = reward_rate(1, 0, switch_idx_2)
                rightRewardRate[0] = reward_rate(2, 0, switch_idx_2)
                leftRewardRate[2] = reward_rate(6, switch_idx_2, n_plot - 1)
                rightRewardRate[2] = reward_rate(5, switch_idx_2, n_plot - 1)
            else:
                leftRewardRate[0] = reward_rate(1, 0, switch_idx_1)
                rightRewardRate[0] = reward_rate(2, 0, switch_idx_1)
                leftRewardRate[1] = reward_rate(3, switch_idx_1, switch_idx_2)
                rightRewardRate[1] = reward_rate(4, switch_idx_1, switch_idx_2)
                leftRewardRate[2] = reward_rate(6, switch_idx_2, n_plot - 1)
                rightRewardRate[2] = reward_rate(5, switch_idx_2, n_plot - 1)

        label_x = max(1, n_plot - 100)
        if switch_idx_1 is None and switch_idx_2 is None:
            ax.text(label_x, -2, f'{leftRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(label_x, 2, f'{rightRewardRate[0]:.2f}', fontsize=12, color='red')
        elif switch_idx_2 is None:
            ax.text(switch_idx_1 - 100, -3, f'{leftRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(switch_idx_1 - 100, 3, f'{rightRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 200), -3, f'{leftRewardRate[1]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 200), 3, f'{rightRewardRate[1]:.2f}', fontsize=12, color='red')
        elif switch_idx_1 is None:
            ax.text(switch_idx_2 - 100, -3, f'{leftRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(switch_idx_2 - 100, 3, f'{rightRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 200), 3, f'{leftRewardRate[2]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 200), -3, f'{rightRewardRate[2]:.2f}', fontsize=12, color='red')
        else:
            ax.text(switch_idx_1 - 100, -4, f'{leftRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(switch_idx_1 - 100, 4, f'{rightRewardRate[0]:.2f}', fontsize=12, color='red')
            ax.text(switch_idx_2 - 100, -4, f'{leftRewardRate[1]:.2f}', fontsize=12, color='red')
            ax.text(switch_idx_2 - 100, 4, f'{rightRewardRate[1]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 100), -4, f'{leftRewardRate[2]:.2f}', fontsize=12, color='red')
            ax.text(max(1, n_plot - 100), 4, f'{rightRewardRate[2]:.2f}', fontsize=12, color='red')

        ax.set_xlim([0, n_plot])
        ax.set_ylim([-max_val - 0.5, max_val + 0.5])

        if max_val == 1:
            ax.set_yticks([-1.25, -1, 1, 1.25])
            ax.set_yticklabels(['Reward', 'A', 'B', 'Reward'])
        elif max_val == 2:
            if switch_idx_1 is not None:
                ax.axvline(switch_idx_1 + 1, color='black', linewidth=3)
            ax.set_yticks([-2.25, -2, -1, 1, 2, 2.25])
            ax.set_yticklabels(['Reward', 'A', 'C', 'B', 'D', 'Reward'])
        elif max_val == 3:
            if switch_idx_1 is not None:
                ax.axvline(switch_idx_1 + 1, color='black', linewidth=3)
            if switch_idx_2 is not None:
                ax.axvline(switch_idx_2 + 1, color='black', linewidth=3)
            ax.set_yticks([-3.25, -3, -2, -1, 1, 2, 3, 3.25])
            ax.set_yticklabels(['Reward', 'A', 'C', 'D', 'B', 'D', 'C', 'Reward'])

        ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax2 = axs[1]
        window_size = 60
        runningReward = np.full((n_plot, 2), np.nan)

        for ii in range(max(0, n_plot - window_size)):
            end_incl = ii + window_size - 1
            if switch_idx_1 is None and switch_idx_2 is None:
                runningReward[ii, 0] = reward_rate(1, ii, end_incl)
                runningReward[ii, 1] = reward_rate(2, ii, end_incl)
            else:
                if switch_idx_2 is None:
                    if ii < switch_idx_1:
                        valid_end = min(end_incl, switch_idx_1)
                        runningReward[ii, 0] = reward_rate(1, ii, valid_end)
                        runningReward[ii, 1] = reward_rate(2, ii, valid_end)
                    else:
                        runningReward[ii, 0] = reward_rate(3, ii, end_incl)
                        runningReward[ii, 1] = reward_rate(4, ii, end_incl)
                elif switch_idx_1 is None:
                    if ii < switch_idx_2:
                        valid_end = min(end_incl, switch_idx_2)
                        runningReward[ii, 0] = reward_rate(1, ii, valid_end)
                        runningReward[ii, 1] = reward_rate(2, ii, valid_end)
                    else:
                        runningReward[ii, 0] = reward_rate(6, ii, end_incl)
                        runningReward[ii, 1] = reward_rate(5, ii, end_incl)
                else:
                    if ii < switch_idx_1:
                        valid_end = min(end_incl, switch_idx_1)
                        runningReward[ii, 0] = reward_rate(1, ii, valid_end)
                        runningReward[ii, 1] = reward_rate(2, ii, valid_end)
                    elif ii < switch_idx_2:
                        valid_end = min(end_incl, switch_idx_2)
                        runningReward[ii, 0] = reward_rate(3, ii, valid_end)
                        runningReward[ii, 1] = reward_rate(4, ii, valid_end)
                    else:
                        runningReward[ii, 0] = reward_rate(6, ii, end_incl)
                        runningReward[ii, 1] = reward_rate(5, ii, end_incl)

        ax2.plot(x, runningReward[:, 0], color='red')
        ax2.plot(x, runningReward[:, 1], color='blue')
        ax2.plot([1, n_plot], [0.7, 0.7], color='gray', linestyle='--')
        ax2.set_ylim([0, 1])
        ax2.set_xlim([1, n_plot])
        ax2.set_title('Running average in 60-trial window')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        if max_val == 2 and switch_idx_1 is not None:
            ax2.axvline(switch_idx_1 + 1, color='black', linewidth=3)
        elif max_val == 3:
            if switch_idx_1 is not None:
                ax2.axvline(switch_idx_1 + 1, color='black', linewidth=3)
            if switch_idx_2 is not None:
                ax2.axvline(switch_idx_2 + 1, color='black', linewidth=3)

        fig.tight_layout()
        png_path = os.path.join(save_path, f'{save_name}.png')
        svg_path = os.path.join(save_path, f'{save_name}.svg')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        return fig
    

def read_video(videoPath, frame, ifgray):
    # ifgray: if convert the image to grayscale
    vid = imageio.get_reader(videoPath)

        #for ii in tqdm(range(self.nFrames)):
    if ifgray:
        image = color.rgb2gray(vid.get_data(frame))
    else:
        image = vid.get_data(frame)
        #   [xdim, ydim] = image.shape
        #    if ii == 0:
        #        # get video dimensions
                #imageStack = np.zeros((xdim, ydim, self.nFrames))
        #        imageStack = []
        #    imageStack.append(image)
    return image

def distance_points_to_line(x_coords, y_coords, line_point1, line_point2):
    """
    Calculate the perpendicular distances from multiple points to a line defined by two points.

    Parameters:
    x_coords (array-like): Array of x-coordinates for the points.
    y_coords (array-like): Array of y-coordinates for the points.
    line_point1 (tuple): The first point on the line (x1, y1).
    line_point2 (tuple): The second point on the line (x2, y2).

    Returns:
    np.ndarray: An array of distances from each point to the line.
    """
    x0 = np.array(x_coords)
    y0 = np.array(y_coords)
    x1, y1 = line_point1
    x2, y2 = line_point2

    # Calculate the components of the distance formula
    numerator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Distance from each point to the line
    distances = numerator / denominator
    return distances

def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def _worksheet_to_df(ws):
    values = ws.get_all_values()
    if len(values) == 0:
        return pd.DataFrame()

    header_row = 0
    if not any(str(cell).strip() for cell in values[0]):
        if len(values) > 1:
            header_row = 1
        else:
            return pd.DataFrame()

    headers = [str(cell).strip() for cell in values[header_row]]
    data_rows = values[header_row + 1:]
    return pd.DataFrame(data_rows, columns=headers)

def fetch_rotarod(google_url, root_dir, strains, ages):

    gc = gspread.service_account(filename="credentials/rotarod_reader.json")
    sh = gc.open_by_url(google_url)

    rotarod_ws = sh.worksheet("Rotarod")
    df = _worksheet_to_df(rotarod_ws)

    # convert numeric columns to nullable integer dtype so numeric fills work
    numeric_cols = ['AGE', 'DATE', 'ODOR', 'TRIAL']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # add a sex colum to df
    df['gender'] = ''

    # forward merge for date, start age


    # get some other info from another tab

    id_ws = sh.worksheet('ASD IDs (COMPREHENSIVE)')
    values = id_ws.get_all_values()
    ID_df = pd.DataFrame(values[2:], columns=values[1])

    merge_col = ['ROTAROD START AGE', 'ROTAROD START DATE', 
                    'ODOR START DATE']
    for cc in merge_col:
        ID_df[cc] = ID_df[cc].replace('', np.nan)
        ID_df[cc] = ID_df[cc].ffill()
    # go over df, if column is empty, find if there is proper values in ID_df

    #%% remember to add digging later

    columns2fill = ['AGE', 'DATE', 'ODOR', 'GENOTYPE', 'SEX']
    columns2look = ['ROTAROD START AGE', 'ROTAROD START DATE', 
                    'ODOR START DATE', 'GENOTYPE', 'SEX']
    nEntries = df.shape[0]

    startRow = 408  # records before this row is disregarded
    # due to different exp. protocol

    for ee in tqdm(range(408, nEntries)):
        animal_id = df.loc[ee, 'ASD_ID']
        ID_idx = np.where(ID_df['ASD ID'] == animal_id)[0][0]

        sheet_row = ee + 2  # IMPORTANT
        for cIdx, col in enumerate(columns2fill):
            val = df.loc[ee, col]
            if pd.isna(val) or val == '':  # if the entry is empty
                # try to find the corresponding colomn in ID_df

                # update the corresponding value
                sheet_col = df.columns.get_loc(col) + 1

                if col == 'AGE':
                    if not ID_df.loc[ID_idx, columns2look[cIdx]]=='--':
                        startAge = int(ID_df.loc[ID_idx, columns2look[cIdx]])
                        days2add = int(np.floor((int(df.loc[ee, 'TRIAL']) - 1) / 3))
                        value = startAge + days2add

                elif col == 'DATE':
                    startDate = ID_df.loc[ID_idx, columns2look[cIdx]]
                    if not startDate == '--':
                        dt = datetime.strptime(startDate, '%m/%d/%y')
                        dt2 = dt + timedelta(days=int(np.floor(int((df.loc[ee, 'TRIAL']) - 1) / 3)))
                        value = int(dt2.strftime('%Y%m%d'))

                elif col == 'ODOR':
                    odordt = ID_df.loc[ID_idx, columns2look[cIdx]]
                    if not odordt == '--':
                        dt = datetime.strptime(odordt, '%m/%d/%y')
                        value = int(dt.strftime('%Y%m%d'))
                    else:
                        value = 0

                elif col == 'GENOTYPE':
                    value = ID_df.loc[ID_idx, columns2look[cIdx]]

                elif col == 'SEX': # 
                    value = ID_df.loc[ID_idx, columns2look[cIdx]]

                df.loc[ee, col] = value
                #rotarod_ws.update_cell(sheet_row, sheet_col, value)

    values = [df.columns.tolist()] + df.astype(str).values.tolist()

    #rotarod_ws.clear()
    #rotarod_ws.update(values, "A1")

    rotarod_data_dir = root_dir


    # go over each folder, update the RR_results.csv 
    for ss in strains:
        for aa in ages:
            strain_folder = os.path.join(rotarod_data_dir, f'{ss}_{aa}')
            if not os.path.exists(strain_folder):
                os.makedirs(strain_folder)
            data_folder = os.path.join(strain_folder, 'Data')
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            RR_file = os.path.join(data_folder, 'RR_results.csv')
            animal_file = os.path.join(data_folder, 'AnimalList.csv')
            
            # get the result of a given strain and age from df

            # for 'age', remove empty column with '' first
            age_col = df['AGE']
            age_col[df['AGE']==''] = np.nan

            if aa=='adol':
                df_mask = np.logical_and(df['STRAIN']==ss, age_col<45)
            else:
                df_mask = np.logical_and(df['STRAIN']==ss, age_col>=45)

            df_group = df.loc[df_mask,:]
            
            animals = np.unique(df_group['ASD_ID'])
            genotypes = []
            for animal in animals:
                geno = np.unique(df_group['GENOTYPE'][df_group['ASD_ID']==animal])
                genotypes.append(geno[0])

            # update animalList.csv
            animal_list_df = pd.DataFrame({'AnimalID': animals, 'Genotype': genotypes})
            animal_list_df.to_csv(animal_file, index=False)

            # update RR_Results.csv
            # columns: AnimalID, Genpotype, Age, Weight, Date, Trial, Performance, FBT, Odor, Digging

            # remember to add digging later
            performance_df = pd.DataFrame({'AnimalID': df_group['ASD_ID'], 
                                        'Genotype': df_group['GENOTYPE'],
                                        'Age': df_group['AGE'],
                                        'Gender': df_group['SEX'],
                                        'Weight': df_group['WEIGHT'],
                                        'Date': df_group['DATE'],
                                        'Trial': df_group['TRIAL'],
                                        'Performance': df_group['PERFORMANCE'],
                                        'FBT': df_group['FBT'],
                                        'Odor': df_group['ODOR']
                                        })
            performance_df.to_csv(RR_file, index=False)

def clean_rotarod_videos(video_folder):
    # input
    # video_folder: the folder where the raw videos are stored
    # root_dir: the root directory where the rotarod data is stored
    # strains: list of strains to fetch
    # ages: list of ages to fetch

    # clean up raw files
    # 1. remove size 0 files
    # 2. remove extra spaces in filenames

    video_sessions = os.listdir(video_folder)
    for video_session in video_sessions:
        session_path = os.path.join(video_folder, video_session)
        if not os.path.isdir(session_path):
            continue

        with os.scandir(session_path) as files:
            for file_entry in files:
                if not file_entry.is_file():
                    continue

                file_path = file_entry.path
                if file_entry.stat().st_size == 0:
                    os.remove(file_path)
                    continue

                if ' ' in file_entry.name:
                    os.rename(file_path, os.path.join(session_path, file_entry.name.replace(' ', '')))

    # go through the animalList in root_dir/strains, look for videos in rawvideo files
    
def organize_beh_videos(root_folders, raw_video_folder, dlc_folder):
    # move the videos to the corresponding folders in root_dir
    # find the corresponding dlc files from dlc_folder
    # move videos without DLC files to a separate folder for DLC labeling

    strain_folders = os.listdir(root_folders)
    video_exts = {'.avi', '.mp4', '.mov', '.m4v'}
    dlc_csv_index = []

    # look for DLC csv files in dlc_folder and 
    # create an index for them
    if os.path.isdir(dlc_folder):
        with os.scandir(dlc_folder) as dlc_entries:
            dlc_csv_index = sorted(
                (entry.name.lower(), entry.name, entry.path)
                for entry in dlc_entries
                if entry.is_file() and entry.name.lower().endswith('.csv')
            )

    #%% move exising video recordings to destination folders
    for strain_folder in strain_folders:
        data_folder = os.path.join(root_folders, strain_folder, 'Data')
        # load animalCSV
        animalList = pd.read_csv(os.path.join(data_folder, 'AnimalList.csv'))

        animal_ids = tuple(animalList['AnimalID'].dropna().astype(str).unique())
        for aID in animal_ids:
            behavioral_recordings_folder = os.path.join(data_folder, aID,'Rotarod', 'BehavioralRecording')
            

            with os.scandir(raw_video_folder) as raw_entries:
                for raw_entry in raw_entries:
                    if not raw_entry.is_dir() or not raw_entry.name.startswith(aID):
                        continue

                    destination_path = os.path.join(behavioral_recordings_folder, raw_entry.name)
                    if os.path.exists(destination_path):
                        print(f"Destination already exists, skipping {raw_entry.path}")
                        continue

                    os.makedirs(behavioral_recordings_folder, exist_ok=True)
                    if not move_directory_safely(raw_entry.path, destination_path):
                        continue

    #%% move DLC labels to folders containing matching videos
    for strain_folder in strain_folders:
        data_folder = os.path.join(root_folders, strain_folder, 'Data')
        if not os.path.isdir(data_folder):
            continue

        with os.scandir(data_folder) as animal_entries:
            for animal_entry in animal_entries:
                if not animal_entry.is_dir():
                    continue

                behavioral_recordings_folder = os.path.join(
                    animal_entry.path, 'Rotarod', 'BehavioralRecording'
                )
                if not os.path.isdir(behavioral_recordings_folder):
                    continue

                for folder_path, _, filenames in os.walk(behavioral_recordings_folder):
                    for filename in filenames:
                        video_stem, video_ext = os.path.splitext(filename)
                        if video_ext.lower() not in video_exts:
                            continue

                        prefix = video_stem.lower()
                        start = bisect_left(dlc_csv_index, (prefix, '', ''))
                        end = bisect_right(dlc_csv_index, (prefix + '\uffff', '', ''))

                        for _, dlc_name, dlc_path in dlc_csv_index[start:end]:
                            if not os.path.exists(dlc_path):
                                continue

                            destination_path = os.path.join(folder_path, dlc_name)
                            if os.path.exists(destination_path):
                                print(f"Destination already exists, skipping {dlc_path}")
                                continue

                            shutil.move(dlc_path, destination_path)

    #%% copy videos withoug DLC labeling to a separate folder 
    # for DLC labeling
    forDLCfolder = r'Y:\HongliWang\Rotarod\DLC_training'
    os.makedirs(forDLCfolder, exist_ok=True)

    for strain_folder in strain_folders:
        data_folder = os.path.join(root_folders, strain_folder, 'Data')
        if not os.path.isdir(data_folder):
            continue

        with os.scandir(data_folder) as animal_entries:
            for animal_entry in animal_entries:
                if not animal_entry.is_dir():
                    continue

                behavioral_recordings_folder = os.path.join(
                    animal_entry.path, 'Rotarod', 'BehavioralRecording'
                )
                if not os.path.isdir(behavioral_recordings_folder):
                    continue

                for folder_path, _, filenames in os.walk(behavioral_recordings_folder):
                    csv_names = sorted(
                        filename.lower()
                        for filename in filenames
                        if filename.lower().endswith('.csv')
                    )

                    for filename in filenames:
                        video_stem, video_ext = os.path.splitext(filename)
                        if video_ext.lower() not in video_exts:
                            continue

                        prefix = video_stem.lower()
                        start = bisect_left(csv_names, prefix)
                        has_matching_csv = (
                            start < len(csv_names)
                            and csv_names[start].startswith(prefix)
                        )
                        if has_matching_csv:
                            continue

                        source_path = os.path.join(folder_path, filename)
                        destination_path = os.path.join(forDLCfolder, filename)
                        if os.path.exists(destination_path):
                            print(f"Destination already exists, skipping {source_path}")
                            continue

                        shutil.copy2(source_path, destination_path)


def move_directory_safely(source_path, destination_path):
    """Move a directory to the destination, skipping it if access is denied."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    try:
        shutil.move(source_path, destination_path)
        return True
    except (PermissionError, shutil.Error, OSError) as exc:
        print(f"Skipping {source_path}: {exc}")
        return False


#%% test script
if __name__ == '__main__':
    root_dir = r'Y:\HongliWang\Rotarod\ASD_strains'
    #strains = 'TSC2_adol'
    video_folder = r'Y:\HongliWang\Rotarod\rawRecordings_260622'
    dlc_folder = r'Y:\HongliWang\Rotarod\Filtered_DLC'

    # remove size 0 files and clean the filenames
    clean_rotarod_videos(video_folder)

    # move the videos to the corresponding folders in root_dir
    # find the corresponding dlc files from dlc_folder
    # move videos without DLC files to a separate folder for DLC labeling
    organize_beh_videos(root_dir, video_folder, dlc_folder)   
# %%
 
