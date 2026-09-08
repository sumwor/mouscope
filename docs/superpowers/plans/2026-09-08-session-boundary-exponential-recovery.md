# Session-Boundary Exponential Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a standalone diagnostic that fits exponential post-session recovery at group and animal levels with animal-cluster bootstrap uncertainty.

**Architecture:** One new CLI script imports the established read-only behavioral helpers, constructs boundary-level 300-trial post-session bins, and exposes small pure functions for model evaluation, fitting, aggregation, bootstrap, and plotting. A focused unittest module exercises each unit and the end-to-end synthetic path; real outputs remain outside the repository.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy `least_squares`, Matplotlib, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-08-session-boundary-exponential-recovery-design.md`

## Global Constraints

- Do not modify `analyze_session_boundary.py`, `analyze_boundary_window_sensitivity.py`, `compare_session_aware_learning.py`, `behavioral_pipeline.py`, `utils_model.py`, or behavioral source CSVs.
- Reuse `ReadOnlyBehDataOdor`, `load_protocol_sessions()`, and `rewarded_vector()` from `analyze_session_boundary.py`.
- Use pre=300, post=300, 20-trial non-overlapping bins, and the unchanged first-50 empirical drop definition.
- Fit `P(n) = P_inf - A * exp(-n / lambda)` without forcing P_inf to the empirical baseline or shifting predictions.
- Use P_inf `[0,1]`, A `[0,1]`, lambda `[1,300]`, deterministic multiple starts, and at least eight finite bins.
- Keep the primary group curve boundary-weighted; bootstrap 1,000 animal clusters while preserving boundary weighting.
- Label intervals as “animal-cluster bootstrap 95% CIs” where space permits.
- Make descriptive comparisons only; do not claim inferential WT/HET effects.

---

### Task 1: Exponential model, bin prediction, optimizer, and warnings

**Files:**
- Create: `fit_session_boundary_recovery.py`
- Create: `tests/test_fit_session_boundary_recovery.py`

**Interfaces:**
- Produces: `exponential_recovery(n, p_inf, amplitude, recovery_lambda) -> ndarray`
- Produces: `binned_model_prediction(bin_starts, bin_ends, parameters) -> ndarray`
- Produces: `fit_exponential_recovery(curve, min_bins=8) -> dict`
- Produces: `parameter_warnings(parameters, fit_success) -> dict`

- [ ] **Step 1: Write failing model and optimizer tests**

```python
def test_binned_prediction_is_mean_over_integer_trials():
    got = recovery.binned_model_prediction(
        np.array([0]), np.array([19]), np.array([0.8, 0.2, 75.0])
    )
    expected = np.mean(0.8 - 0.2 * np.exp(-np.arange(20) / 75.0))
    np.testing.assert_allclose(got, [expected])

def test_seeded_synthetic_fit_recovers_parameters():
    rng = np.random.default_rng(240513)
    starts = np.arange(0, 300, 20)
    ends = starts + 19
    expected = recovery.binned_model_prediction(starts, ends, [0.8, 0.2, 75.0])
    curve = pd.DataFrame({"BinStart": starts, "BinEnd": ends,
                          "ObservedMean": expected + rng.normal(0, 0.004, len(starts))})
    fit = recovery.fit_exponential_recovery(curve)
    assert fit["FitSuccess"]
    assert abs(fit["P_inf"] - 0.8) < 0.04
    assert abs(fit["A"] - 0.2) < 0.05
    assert abs(fit["lambda"] - 75) < 30

def test_fit_requires_eight_finite_bins():
    curve = pd.DataFrame({"BinStart": np.arange(0, 140, 20),
                          "BinEnd": np.arange(19, 159, 20),
                          "ObservedMean": np.linspace(0.5, 0.7, 7)})
    fit = recovery.fit_exponential_recovery(curve)
    assert not fit["FitSuccess"]
    assert fit["NFitBins"] == 7

def test_warning_flags_report_each_reason():
    warnings = recovery.parameter_warnings(np.array([0.999, 0.001, 299.5]), True)
    assert warnings == {"PInfWarning": True, "AWarning": True,
                        "LambdaWarning": True, "P0Warning": False,
                        "BoundaryWarning": True}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: import failure because `fit_session_boundary_recovery.py` does not exist.

- [ ] **Step 3: Implement model and deterministic multistart fitting**

```python
PARAMETER_LOWER = np.array([0.0, 0.0, 1.0])
PARAMETER_UPPER = np.array([1.0, 1.0, 300.0])

def exponential_recovery(n, p_inf, amplitude, recovery_lambda):
    return p_inf - amplitude * np.exp(-np.asarray(n) / recovery_lambda)

def binned_model_prediction(bin_starts, bin_ends, parameters):
    p_inf, amplitude, recovery_lambda = parameters
    return np.array([
        exponential_recovery(np.arange(start, end + 1), p_inf,
                             amplitude, recovery_lambda).mean()
        for start, end in zip(bin_starts.astype(int), bin_ends.astype(int))
    ])
```

Implement `fit_exponential_recovery()` with `scipy.optimize.least_squares`, 27 deterministic starts formed from three clipped late-performance values, A values `(0.05, 0.2, 0.5)`, and lambda values `(10, 75, 200)`. Retain the successful finite result with lowest RSS, calculate RMSE and R2, and return NaN parameters for insufficient/failed fits.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add fit_session_boundary_recovery.py tests/test_fit_session_boundary_recovery.py
git commit -m "feat: add exponential recovery fitter"
```

### Task 2: Boundary extraction, empirical metrics, and group/animal curves

**Files:**
- Modify: `fit_session_boundary_recovery.py`
- Modify: `tests/test_fit_session_boundary_recovery.py`

**Interfaces:**
- Consumes: existing behavioral session frames and Task 1 fitter
- Produces: `extract_boundary_data(session_frames) -> tuple[DataFrame, DataFrame]`
- Produces: `aggregate_group_curve(boundary_bins) -> DataFrame`
- Produces: `fit_animal_curves(boundary_bins, boundary_metrics) -> DataFrame`

- [ ] **Step 1: Write failing extraction and weighting tests**

```python
def test_boundary_extraction_preserves_metadata_and_empirical_drops():
    previous = make_session([1, 1, 1, 1, 0] * 60, 1, "20230101")
    following = make_session([1] * 30 + [0] * 20 + [1] * 250, 2, "20230102")
    bins, metrics = recovery.extract_boundary_data([previous, following])
    row = metrics.iloc[0]
    assert row["PreBoundaryBaseline"] == 0.8
    assert row["InitialPostPerformance"] == 0.6
    assert row["AbsoluteDrop"] == row["DropMagnitude"]
    assert np.isclose(row["PercentDrop"], 25.0)
    assert list(bins["BinStart"]) == list(range(0, 300, 20))

def test_group_curve_is_boundary_weighted_not_animal_equal():
    rows = pd.DataFrame({"Animal": [1, 1, 2], "Protocol": ["AB"] * 3,
                         "Genotype": ["WT"] * 3, "BoundaryNumber": [1, 2, 1],
                         "BinStart": [0] * 3, "BinEnd": [19] * 3,
                         "Performance": [0.0, 0.0, 1.0]})
    curve = recovery.aggregate_group_curve(rows)
    assert np.isclose(curve.iloc[0]["ObservedMean"], 1 / 3)
    assert curve.iloc[0]["NBoundaries"] == 3
    assert curve.iloc[0]["NAnimals"] == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: failures for missing extraction and aggregation functions.

- [ ] **Step 3: Implement boundary and curve construction**

Construct one metrics row per valid consecutive-session boundary and one bin row
per observed post bin. Calculate `PercentDrop` on each boundary before any group
or animal aggregation, using `1e-12` as the zero-baseline tolerance. Group by
Protocol, Genotype, and BinStart for boundary-level mean/SEM/counts. Animal curves
group by Animal, Protocol, and BinStart before calling the Task 1 fitter.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: all Task 1–2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add fit_session_boundary_recovery.py tests/test_fit_session_boundary_recovery.py
git commit -m "feat: build boundary recovery datasets"
```

### Task 3: Animal-cluster bootstrap and group summaries

**Files:**
- Modify: `fit_session_boundary_recovery.py`
- Modify: `tests/test_fit_session_boundary_recovery.py`

**Interfaces:**
- Consumes: boundary-bin rows and Task 1 fitter
- Produces: `bootstrap_group_fit(group_bins, n_replicates, rng) -> tuple[dict, ndarray]`
- Produces: `fit_group_curves(boundary_bins, boundary_metrics, n_replicates, seed) -> tuple[DataFrame, DataFrame]`

- [ ] **Step 1: Write failing cluster-bootstrap test**

```python
def test_bootstrap_resamples_whole_animal_clusters_with_multiplicity():
    sample = recovery.resample_animal_clusters(
        make_cluster_rows(), np.random.default_rng(7), sampled_animals=np.array([1, 1])
    )
    assert sample["Animal"].nunique() == 1
    assert len(sample) == 2 * len(make_cluster_rows().query("Animal == 1"))
    assert sample["BootstrapCluster"].nunique() == 2
```

Add a deterministic bootstrap integration test using 20 replicates and asserting
successful count, ordered percentile intervals, fixed-seed repeatability, and a
curve-band array of shape `(300, 2)`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: failures for missing cluster bootstrap functions.

- [ ] **Step 3: Implement cluster resampling and group fitting**

Resample unique animal identifiers with replacement. For every draw, copy all
boundary-bin rows belonging to that animal and attach a distinct bootstrap-cluster
identifier so repeated draws remain repeated clusters. Reconstruct the boundary-
weighted group mean, refit, retain successful parameters, and calculate percentile
intervals and trial-level bands.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: all Task 1–3 tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add fit_session_boundary_recovery.py tests/test_fit_session_boundary_recovery.py
git commit -m "feat: add animal-cluster recovery bootstrap"
```

### Task 4: CSV schemas, figures, README, and CLI

**Files:**
- Modify: `fit_session_boundary_recovery.py`
- Modify: `tests/test_fit_session_boundary_recovery.py`

**Interfaces:**
- Consumes: group/animal summaries, aligned curves, and bootstrap bands
- Produces: three requested CSVs, four requested PNGs, and `README_session_boundary_recovery_fit.txt`
- Produces: `main()` CLI and `run_synthetic_check()`

- [ ] **Step 1: Write failing artifact and synthetic CLI tests**

Create a small synthetic boundary dataset and assert:

```python
with tempfile.TemporaryDirectory() as directory:
    recovery.write_outputs(group_summary, animal_summary, curve, bins, Path(directory))
    expected = {
        "recovery_group_fit_summary.csv", "recovery_animal_fit_summary.csv",
        "recovery_group_aligned_curve.csv", "AB_exponential_recovery_fit.png",
        "CD_exponential_recovery_fit.png", "recovery_parameter_comparison.png",
        "fitted_vs_empirical_drop.png", "README_session_boundary_recovery_fit.txt",
    }
    assert expected <= {path.name for path in Path(directory).iterdir()}
    assert "animal-cluster bootstrap 95% CI" in (
        Path(directory) / "README_session_boundary_recovery_fit.txt"
    ).read_text()
```

Test `run_synthetic_check()` returns a successful fit within P_inf `0.04`, A
`0.05`, and lambda `30` of `0.80`, `0.20`, and `75`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: failures for missing output and CLI functions.

- [ ] **Step 3: Implement artifacts and command-line execution**

Implement the requested group-fit plots, animal parameter plots, fitted-versus-
empirical plots, README, output-directory protection, arguments, real-data loop,
and concise terminal table. Figure annotations and the README use “animal-cluster
bootstrap 95% CI(s)” explicitly where space permits.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: all tests pass without warnings or errors.

- [ ] **Step 5: Commit Task 4**

```bash
git add fit_session_boundary_recovery.py tests/test_fit_session_boundary_recovery.py
git commit -m "feat: complete recovery fit diagnostics"
```

### Task 5: Compile, real-data execution, and evidence-based interpretation

**Files:**
- Verify: `fit_session_boundary_recovery.py`
- Verify: `/Users/anatta/Documents/GitHub/session_boundary_recovery_fit_output/*`

**Interfaces:**
- Consumes: completed CLI and specified behavioral dataset
- Produces: validated real-data artifacts and concise descriptive findings

- [ ] **Step 1: Run complete unit suite**

Run: `python -m unittest tests.test_fit_session_boundary_recovery -v`

Expected: zero failures and zero errors.

- [ ] **Step 2: Compile the standalone script**

Run: `python -m py_compile fit_session_boundary_recovery.py`

Expected: exit code 0.

- [ ] **Step 3: Run the synthetic parameter-recovery check**

Run: `python fit_session_boundary_recovery.py --synthetic-test-only`

Expected: successful approximate recovery of P_inf=0.80, A=0.20, lambda=75.

- [ ] **Step 4: Snapshot protected source and input state**

Record Git status and SHA-256 checksums for the behavioral index/source CSVs and
the protected existing Python files before the real run.

- [ ] **Step 5: Execute 1,000-replicate real analysis**

```bash
python fit_session_boundary_recovery.py \
  --root-dir "/Users/anatta/Documents/Hongli/Juvi_ASD Deterministic/TSC2_adol" \
  --output-dir "/Users/anatta/Documents/GitHub/session_boundary_recovery_fit_output" \
  --strain "TSC2_adol" \
  --bootstrap-replicates 1000
```

Expected: all eight artifacts written and a four-row group table printed.

- [ ] **Step 6: Validate outputs and protected inputs**

Assert exact output schemas, four group rows, expected animal rows, at least eight
fit bins per successful fit, composite warning equivalence, ordered finite CIs,
bootstrap counts no greater than 1,000, and bin values in `[0,1]`. Recompute fit
metrics from curve rows where possible and compare protected-file checksums.

- [ ] **Step 7: Visually inspect every PNG**

Open all four figures and confirm axes, observed SEM, fitted curves, annotations,
legends, individual animal points, summary markers, and non-overlapping labels.

- [ ] **Step 8: Report descriptive findings**

Print and hand off Protocol, Genotype, NAnimals, NBoundaries, P_inf with
animal-cluster bootstrap 95% CI, A with the same CI label, lambda with the same CI
label, RMSE, and R2. State fit quality and directional comparisons separately
from inferential claims, and explicitly discuss warning/constrained fits.

