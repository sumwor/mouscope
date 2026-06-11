# utility functions for behaviral analysis
import csv
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f
from scipy.stats import chi2
import os
import shutil
import imageio
from skimage import color

import matplotlib
matplotlib.use('QtAgg') 

import matplotlib.pyplot as plt
plt.ion()
import matplotlib.pyplot as plt
# Deeplabcut related, and MotionSequence related functions
from scipy.signal import butter, filtfilt

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
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9
        )

    ax.set_xticks(np.arange(len(trial_order)))
    ax.set_xticklabels([str(t) for t in trial_order])
    ax.set_xlabel('Trial / Block')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if summary_path is not None:
        os.makedirs(os.path.join(summary_path, 'BehPlots'), exist_ok=True)
        fig.savefig(os.path.join(summary_path, 'BehPlots', f'{save_name}.png'), dpi=300)
        fig.savefig(os.path.join(summary_path, 'BehPlots', f'{save_name}.svg'), format='svg')

    #return fig, ax, stats_df, clean_df

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

#%%
if __name__ == '__main__':

    #Example usage
    root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2\Analysis'
    # find the folder in the root_dir
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    for f in folders:
        #if not os.path.exists(os.path.join(root_dir, f, 'Odor')):
        #os.makedirs(os.path.join(root_dir, f, 'Odor'), exist_ok=True)
        # move behavior directory and every file inside in to the Odor folder
        # remove .csv files in the folder
        sub_root = os.path.join(root_dir, f, 'Odor','Behavior')
        sub_folders = [ff for ff in os.listdir(sub_root) if os.path.isdir(os.path.join(sub_root, ff))]
        for folder in sub_folders:
            for file in os.listdir(os.path.join(sub_root, folder)):
                if file.endswith('.csv'):
                    os.remove(os.path.join(sub_root, folder, file))

        
# %%
 
