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
#matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()

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
from scipy.io import loadmat
import statsmodels.api as sm
from scipy.stats import chi2, shapiro
from rpy2 import robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

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

def run_learning_glmm(perf_df, behavior, save_name, summary_path):
    """Fit frequentist lme4 learning models with a subject random intercept."""


    data = perf_df.copy()
    is_rotarod = behavior=='rotarod'
    outcome = 'performance'
    required = ['subject', 'genotype', 'trial', outcome]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    data[outcome] = pd.to_numeric(data[outcome], errors='coerce')
    data['trial'] = pd.to_numeric(data['trial'], errors='coerce')
    data = data.dropna(subset=required).copy()
    if data['genotype'].nunique() < 2 or data['trial'].nunique() < 2:
        return pd.DataFrame({'error': ['Insufficient genotype or trial variation']}), None, None

    levels = sorted(data['genotype'].astype(str).unique())
    reference = 'WT' if 'WT' in levels else levels[0]
    if is_rotarod:
        if (data[outcome] > 0).all() and len(data) >= 3:
            sample = data[outcome].sample(min(len(data), 5000), random_state=0)
            if shapiro(np.log(sample)).pvalue > shapiro(sample).pvalue:
                data['_model_performance'] = np.log(data[outcome])
                distribution = 'lognormal'
            else:
                data['_model_performance'] = data[outcome]
                distribution = 'gaussian'
        else:
            data['_model_performance'] = data[outcome]
            distribution = 'gaussian'
        response = '_model_performance'
    else:
        successes = np.rint(data[outcome] * 100).astype(int)
        if ((data[outcome] < 0) | (data[outcome] > 1) |
                ~np.isclose(data[outcome] * 100, successes)).any():
            raise ValueError('Odor-task performance must be a 0--1 rate from 100 trials.')
        data['successes'] = successes
        data['failures'] = 100 - successes
        response = 'cbind(successes, failures)'
        distribution = 'binomial'

    fixed = {
        'no_genotype_model': f'{response} ~ trial',
        'genotype_model': f'{response} ~ genotype + trial',
        'no_learning_model': f'{response} ~ genotype',
        'full_model': f'{response} ~ genotype * trial',
    }
    formulas = {key: value + ' + (1 | subject)' for key, value in fixed.items()}
    lme4 = importr('lme4')
    stats = importr('stats')
    with localconverter(default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
    ro.globalenv['learning_glmm_data'] = r_data
    ro.globalenv['learning_glmm_reference'] = reference
    ro.r('''
        learning_glmm_data$subject <- factor(learning_glmm_data$subject)

        learning_glmm_data$genotype <- relevel(
            factor(
                as.character(learning_glmm_data$genotype),
                ordered = FALSE
            ),
            ref = learning_glmm_reference
        )
    ''')
    r_data = ro.globalenv['learning_glmm_data']

    def fit(formula):
        if distribution == 'binomial':
            return lme4.glmer(ro.Formula(formula), data=r_data, family=stats.binomial())
        return lme4.lmer(ro.Formula(formula), data=r_data, REML=False)

    models = {key: fit(formula) for key, formula in formulas.items()}

    def lr_test(reduced, full, term):
        comparison = ro.r['anova'](reduced, full, test='Chisq')
        with localconverter(default_converter + pandas2ri.converter):
            comparison = ro.conversion.rpy2py(comparison)
        row = comparison.iloc[-1]
        return {'term': term, 'lr_stat': float(row['Chisq']),
                'df': int(row['Df']), 'p_value': float(row['Pr(>Chisq)'])}

    # do one full model only
    full_model_fit = lme4.glmer(
        formulas['full_model'],
        data=r_data,
        family=stats.binomial()
    )
    summary = ro.r['summary'](full_model_fit)
    #print(summary)

    coef_table = summary.rx2('coefficients')

    with localconverter(default_converter + pandas2ri.converter):
        coef_table = ro.conversion.rpy2py(coef_table)

    coef_table = pd.DataFrame(
        coef_table,
            index=list(summary.rx2('coefficients').rownames),
            columns=list(summary.rx2('coefficients').colnames)
        )

    stats_df = pd.DataFrame([
        {
            'term': 'genotype',
            'estimate': coef_table.loc['genotypeHET', 'Estimate'],
            'p_value': coef_table.loc['genotypeHET', 'Pr(>|z|)']
        },
        {
            'term': 'trial',
            'estimate': coef_table.loc['trial', 'Estimate'],
            'p_value': coef_table.loc['trial', 'Pr(>|z|)']
        },
        {
            'term': 'genotype:trial',
            'estimate': coef_table.loc['genotypeHET:trial', 'Estimate'],
            'p_value': coef_table.loc['genotypeHET:trial', 'Pr(>|z|)']
        }
    ])
    # stats_df = pd.DataFrame([
    #     lr_test(models['no_genotype_model'], models['genotype_model'], 'genotype'),
    #     lr_test(models['no_learning_model'], models['genotype_model'], 'trial'),
    #     lr_test(models['genotype_model'], models['full_model'], 'genotype:trial'),
    # ])
    # stats_df['distribution'] = distribution
    # stats_df['n_observations'] = len(data)
    # stats_df['n_subjects'] = data['subject'].nunique()
    # stats_df['n_trials'] = data['trial'].nunique()
    # stats_df['genotype_reference'] = reference
    # stats_df['genotype_levels'] = ','.join(levels)
    # stats_df['repeated_measure_term'] = 'subject_random_intercept'
    if summary_path is not None:
        os.makedirs(summary_path, exist_ok=True)
        stats_df.to_csv(os.path.join(summary_path, f'{save_name} performance glmm.csv'), index=False)
    return stats_df, models, data

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

    if trial_col == 'Block':
        beh = 'odor'
    elif trial_col == 'Trial':
        beh = 'rotarod'

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

    stats_df, _, _ = run_learning_glmm(clean_df, beh, save_name=save_name,summary_path=summary_path)
    #stats_df, _, _ = run_learning_FDA(clean_df, save_name=save_name,summary_path=summary_path)

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
            f"GLMM p genotype = {p_values.get('genotype', np.nan):.3g}\n"
            f"GLMM p learning = {p_values.get('trial', np.nan):.3g}\n"
            f"GLMM p interaction = {p_values.get('genotype:trial', np.nan):.3g}"
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
    
def organize_beh_videos(root_folders, raw_video_folder, dlc_folder, dlc_labeled_folder):
    # move the videos to the corresponding folders in root_dir
    # find the corresponding dlc files from dlc_folder
    # move videos without DLC files to a separate folder for DLC labeling

    strain_folders = [
        f for f in os.listdir(root_folders)
        if os.path.isdir(os.path.join(root_folders, f))
    ]
    video_exts = {'.avi', '.mp4', '.mov', '.m4v'}
    dlc_csv_index = []

    # look for DLC csv files in dlc_folder and 
    # create an index for them
    if os.path.isdir(dlc_labeled_folder):
        if 'DLC' in dlc_labeled_folder: # for deeplabcut labels
            with os.scandir(dlc_labeled_folder) as dlc_entries:
                dlc_csv_index = sorted(
                    (entry.name.lower(), entry.name, entry.path)
                    for entry in dlc_entries
                    if entry.is_file() and entry.name.lower().endswith('filtered.csv')
                )
        elif 'litPose' in dlc_labeled_folder: # for litPose labels
            with os.scandir(dlc_labeled_folder) as dlc_entries:
                dlc_csv_index = sorted(
                    (entry.name.lower(), entry.name, entry.path)
                    for entry in dlc_entries
                    if entry.is_file() and not entry.name.lower().endswith('temporal_norm.csv')
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

#%% code for extracting odor behavior
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
    
    #warnings.filterwarnings('ignore')
    
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
 
