# Session-Boundary Exponential Recovery Design

## Purpose

Create `fit_session_boundary_recovery.py`, a standalone, read-only diagnostic
that models the transient post-session reset with

`P(n) = P_inf - A * exp(-n / lambda)`

for zero-based post-boundary trial `n`. The analysis is descriptive. It compares
AB WT, AB HET, CD WT, and CD HET without treating descriptive genotype
differences as inferential evidence.

The implementation must not modify `analyze_session_boundary.py`,
`analyze_boundary_window_sensitivity.py`, `compare_session_aware_learning.py`,
`behavioral_pipeline.py`, `utils_model.py`, or behavioral source CSVs.

## Inputs and established behavior

The script accepts:

- `--root-dir`
- `--output-dir`
- `--strain`
- `--bootstrap-replicates`, defaulting to `1000`
- `--bootstrap-seed`, with a fixed deterministic default
- `--synthetic-test-only`

The real-data invocation uses:

- root: `/Users/anatta/Documents/Hongli/Juvi_ASD Deterministic/TSC2_adol`
- output: `/Users/anatta/Documents/GitHub/session_boundary_recovery_fit_output`
- strain: `TSC2_adol`

The script imports `ReadOnlyBehDataOdor`, `load_protocol_sessions()`, and
`rewarded_vector()` from `analyze_session_boundary.py`. This preserves the
existing AB/CD inclusion rules, action filtering, CD schedule restriction,
reward conversion, boundary ordering, and source-data protections.

## Boundary-level representation

The primary window is the last 300 trials of the previous session and the first
300 trials of the next session. Each valid consecutive-session boundary is
aligned to post-boundary trial zero.

For every valid boundary the script retains:

- Animal, Genotype, Gender, and Protocol
- BoundaryNumber
- PreviousSession and NextSession
- PreviousDate and NextDate
- PreviousSessionIndex and NextSessionIndex when available
- PreviousProtocolDay and NextProtocolDay when available
- number of available pre- and post-boundary trials
- PreBoundaryBaseline
- InitialPostPerformance
- DropMagnitude and AbsoluteDrop
- PercentDrop and PercentDropValid

PreBoundaryBaseline is the mean rewarded fraction across up to 300 available
pre-boundary trials. InitialPostPerformance is the mean across the first 50
available post-boundary trials. DropMagnitude and AbsoluteDrop are the identical
absolute proportion difference:

`PreBoundaryBaseline - InitialPostPerformance`

PercentDrop is calculated independently for every boundary before aggregation:

`100 * AbsoluteDrop / PreBoundaryBaseline`

If the baseline is non-finite or has absolute value at most `1e-12`, PercentDrop
is NaN and PercentDropValid is false.

Post-boundary rewarded trials are reduced to non-overlapping bins covering
0–19, 20–39, ..., 280–299. Short sessions contribute only observed trials.
There is no smoothing, normalization, vertical shifting, or imputation.

## Group trajectory estimand

For each Protocol × Genotype × post-bin combination, observed performance is
the mean of boundary-level bin performances. SEM is also calculated across
boundary-level bin performances. Therefore, the primary group trajectory is
explicitly boundary-weighted: animals with more valid boundaries contribute
more boundary observations.

This weighting is part of the requested estimand and must not be silently
changed to equal-animal weighting.

## Exponential model and bin prediction

The continuous trial-level model is:

`P(n) = P_inf - A * exp(-n / lambda)`

with bounds:

- P_inf: `[0, 1]`
- A: `[0, 1]`
- lambda: `[1, 300]`

P_inf is estimated exclusively from the post-boundary trajectory and is not
forced to equal PreBoundaryBaseline. A is the fitted recovery amplitude between
P(0) and P_inf. Lambda is the recovery timescale in trials.

Because each observation is a 20-trial mean, fitting compares it with the mean
of model predictions over the corresponding integer trials, rather than with a
single midpoint prediction. Smooth plotting curves evaluate the same fitted
model at every integer trial from 0 through 299.

Fits use unweighted residual least squares over finite observed bin means.
Observed SEM is displayed but is not used as inverse-variance fit weighting.

## Optimization

Use bounded SciPy least-squares optimization with deterministic multiple starts.
Starts cross:

- P_inf values around clipped late-trajectory performance
- low, moderate, and high A values
- short, moderate, and long lambda values

The lowest-RSS successful finite fit is retained. A fit requires at least eight
finite bins. Diagnostics are calculated over the fitted bins:

- RSS
- RMSE
- R2, NaN when total observed variance is numerically zero
- optimizer success and message
- number of fit bins

Failed fits retain identifying metadata, counts, and failure diagnostics while
biological parameters remain NaN.

## Parameter warning flags

Near-bound behavior means being within 1% of a parameter's allowed range.
Every successful fit reports separate Boolean flags:

- `PInfWarning`: P_inf is near 0 or 1
- `AWarning`: A is near 0 or 1
- `LambdaWarning`: lambda is near 1 or 300
- `P0Warning`: fitted `P(0) = P_inf - A` lies outside `[0, 1]`

`BoundaryWarning` is true when any separate flag is true. Failed fits have
`BoundaryWarning=true`; the individual parameter flags remain false unless a
finite fitted parameter specifically triggers them. The optimizer status and
message distinguish failure from parameter-bound behavior.

## Animal-level fits

For each Animal × Protocol, boundary-level bin performances are averaged at
each aligned bin. The exponential model is fit to this animal mean trajectory
when at least eight finite bins are present.

Each row includes:

- Animal, Protocol, Genotype, and Gender
- P_inf, A, lambda, and fitted P(0)
- RSS, RMSE, R2, FitSuccess, optimizer message
- PInfWarning, AWarning, LambdaWarning, P0Warning, BoundaryWarning
- NBoundaries and NFitBins
- mean and median empirical PreBoundaryBaseline
- mean and median empirical DropMagnitude/AbsoluteDrop
- mean and median empirical PercentDrop across valid boundary percentages
- NPercentDropValid

The CSV contains one row for every observed Animal × Protocol combination,
including rows that fail the fit or have insufficient finite bins.

## Animal-cluster bootstrap

For each Protocol × Genotype group, bootstrap animals with replacement using a
fixed seed and 1,000 replicates by default. A sampled animal brings all of its
valid boundaries into the replicate. If sampled multiple times, its complete
boundary cluster is included with the corresponding multiplicity.

Within each replicate, the group curve is reconstructed as the mean over the
resampled boundary rows and refit. Thus the bootstrap is clustered at the
animal level while preserving the primary boundary-weighted group estimand. It
does not switch to equal-animal weighting.

Percentile 95% intervals use successful finite bootstrap fits for P_inf, A, and
lambda. Trial-level 2.5th and 97.5th percentiles across successful fitted curves
form the optional confidence band. The summary reports requested replicate
count, successful count, and success fraction. If no bootstrap fit succeeds,
parameter intervals and the curve band remain NaN.

## Output tables

### `recovery_group_fit_summary.csv`

One row per Protocol × Genotype, containing:

- Protocol, Genotype, NAnimals, NBoundaries
- P_inf, A, lambda, and fitted P(0)
- P_inf_CI_low/high, A_CI_low/high, lambda_CI_low/high
- RSS, RMSE, R2, FitSuccess, optimizer message, NFitBins
- PInfWarning, AWarning, LambdaWarning, P0Warning, BoundaryWarning
- NBootstrapRequested, NBootstrapSuccessful, BootstrapSuccessFraction
- empirical baseline, absolute-drop, and percent-drop summaries

### `recovery_animal_fit_summary.csv`

One row per Animal × Protocol with the animal-fit and empirical fields described
above.

### `recovery_group_aligned_curve.csv`

One row per Protocol × Genotype × post-boundary bin, containing:

- bin start, end, and display center
- observed mean, SEM, boundary count, and animal count
- fitted bin mean
- fitted smooth value at the display center
- bootstrap fitted-curve CI values at the display center

## Figures

### `AB_exponential_recovery_fit.png` and `CD_exponential_recovery_fit.png`

Each protocol figure shows WT and HET observed 20-trial group means with SEM,
the trial-level fitted exponential curve, an optional animal-bootstrap confidence
band, a boundary marker at x=0, axes spanning trials 0–300 and probability 0–1,
and annotations for A and lambda with 95% intervals, N animals, and N boundaries.

### `recovery_parameter_comparison.png`

Three panels compare valid, non-warning animal-level A, lambda, and P_inf values
across AB WT, AB HET, CD WT, and CD HET. Individual animals are shown with
deterministic jitter. Median and IQR are overlaid without bars. Warning or failed
fits are excluded from biological summaries and their counts are disclosed.

### `fitted_vs_empirical_drop.png`

Animal-level valid fitted A is compared with mean empirical AbsoluteDrop and
mean empirical PercentDrop. Separate panels avoid treating absolute proportions
and relative percentages as the same unit. Identity/reference structure is
included only where units match, and points are labeled or colored by group.

## Interpretation and identifiability

The final terminal output reports Protocol, Genotype, NAnimals, NBoundaries,
P_inf and its interval, A and its interval, lambda and its interval, RMSE, and R2.
It then gives descriptive answers about fit quality, AB/CD reset magnitude, CD
HET reset magnitude and lambda, and directional agreement with empirical drops.
No statistically significant genotype claim is made.

Material identifiability limitations are reported in the README and final
interpretation:

- P_inf, A, and lambda trade off when the trajectory does not plateau by trial
  299.
- Flat or noisy curves can drive A near zero and leave lambda weakly identified.
- Each fit has at most 15 binned observations.
- A is partly extrapolated to trial zero because the first observed response is
  averaged across trials 0–19.
- Independent P_inf and A bounds can allow P(0) below zero; this is flagged and
  never clipped or shifted.
- Fitted A measures recovery toward the fitted post-session asymptote, whereas
  empirical DropMagnitude compares the previous-session baseline with the first
  50 post trials. Directional agreement does not imply numerical equivalence.
- Bootstrap intervals can be skewed or truncated by fit bounds and are not
  genotype-effect tests.

## README

`README_session_boundary_recovery_fit.txt` documents the model, fitting window,
binning, optimizer, minimum-bin rule, warnings, bootstrap seed and replicate
count, all identifiability caveats, and interpretation constraints. It states
prominently that the group trajectory is boundary-weighted and the bootstrap
resamples animal clusters while preserving boundary weighting.

## Validation

Implementation follows test-driven development and includes:

1. A seeded synthetic trajectory with P_inf=0.80, A=0.20, and lambda=75 plus
   small noise, with explicit approximate parameter tolerances.
2. Tests for bin-mean model evaluation, multi-start fit selection, insufficient
   bins, fit metrics, warning flags, animal-level pooling, boundary-weighted group
   aggregation, clustered bootstrap multiplicity, output schema, and figure
   generation.
3. `python -m py_compile fit_session_boundary_recovery.py`.
4. Real-data execution with 1,000 bootstrap replicates.
5. Post-run checks for expected rows, finite values, warning consistency, CI
   ordering, bootstrap counts, plot readability, and unchanged source files.
