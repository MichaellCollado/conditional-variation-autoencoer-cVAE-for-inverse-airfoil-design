# Pipeline manual

This repository holds the code, the seed inputs, the published build artifacts
and the run procedure for the study reported in `PAPER.md`, "An avian derived
shape prior as a flag gated objective term in a conditional VAE for inverse
airfoil design". The study trains a conditional VAE for inverse airfoil design
with an avian derived shape prior applied as a flag gated term in the training
objective, and it measures one paired outcome against the same model with the
prior switched off.

`PAPER.md` is the article of record and the authority for this repository. Where
this manual and the article disagree, the article is right and this manual is
wrong.

Two numbering schemes appear below and they are kept apart. A reference to the
article always names it: "Table A3", "Table 8", "Figure 3", "section 2.5 of the
article". A bare section number, like "section 6.4", is a section of this
manual.

This manual walks from a fresh clone to the reported numbers and figures. It
assumes Python and nothing about the study.

Every runtime quoted here comes from **Table A3** of the article, or from the one
figure section 2.5 states. Where the article records no runtime, this manual says
so instead of estimating.

**What you need beyond this clone is XFOIL.** Everything else is here.

---

## 1. What this pipeline is, in one paragraph

Five seed airfoil sections define a design space. A bounded perturbation sampler
draws a population from them. XFOIL labels each shape with its maximum lift to
drag ratio over a fixed nine point angle sweep. A conditional VAE learns to
generate a shape given a requested efficiency, with a conditioning block that
carries a flag. When the flag is set, a prior term pulls the generated shape
toward a region around the avian reference. When the flag is clear, that term is
exactly zero. Everything else in the objective is blind to the flag. Generation
then runs in matched pairs, one shape per arm from a shared latent code, and each
pair is solved and compared.

The reported outcome is the paired difference in target satisfaction error
between the two arms.

---

## 2. Prerequisites

### 2.1 What is published, and what the repository cannot rebuild

The retained drivers begin at build step B16. Every artifact they read is built
at a step below B16, and no retained driver builds any of them. The pre-B16
drivers were deleted from the build after their falsification checks passed, and
no filename is stated for any of them.

**Those artifacts are published, so this is not a problem you have to solve.**
They are in `artifacts/`, ten files and 1,212,369 bytes, committed to this
repository. `artifacts/README.md` records what each one is, which build step
produced it, which driver reads it, and a SHA-256 for each.

```text
population.npz          dataset.npz            normalization.npz
standardization.npz     avian_signature.npz    flag_assignment.npz
conditioning.npz        split.npz              surrogate_ensemble.pt
b07_labelling_progress.jsonl
```

What the repository still cannot do is re-derive those ten files from `seeds/`
alone. The modules that do the work of every pre-B16 step are all here —
`geometry.py` fits the seeds, `dataset.py` draws the population, assembles the
conditioning and splits the data, `prior.py` derives the region extent,
`model.py` trains and freezes the surrogate. What is missing is the driver
scripts that called them in order and wrote the files out. Rebuilding them would
mean rewriting those drivers from the parameter record in `params.py`.

So: the code that produced every published artifact is in this repository, and
the scripts that ran it are not. Everything from B16 onward runs.

### 2.2 Machine and operating system

The reported build ran on one machine and the article claims nothing beyond that.

| Item | Value, from Table A3 |
|---|---|
| CPU | 11th Gen Intel Core i5-11400H, 6 physical cores, 12 logical, 2.70 GHz base |
| Memory | 15.74 GB |
| Operating system | Microsoft Windows 11 Home Single Language, 10.0.26200, build 26200 |
| GPU used | None. The installed torch is the CPU build and CUDA is unavailable |

The solver binary the build calls is a Win32 executable. No portability claim is
made anywhere in the article, and none is made here. Running on any other machine
or operating system is untested.

### 2.3 Software

| Item | Value, from Table A3 |
|---|---|
| Python | 3.14.6 |
| numpy | 2.5.1 |
| torch | 2.13.0+cpu |
| matplotlib | 3.11.1, used by the figure driver alone and by no pipeline module |

`requirements.txt` pins these four and the rest of the environment the reported
runs were made in. The article records the four above; the remaining pins are
read from that environment, and the file says which is which.

No scipy is installed and none should be. The two-sample Kolmogorov-Smirnov
statistic is written out by hand in `dataset.ks_two_sample` for that reason. If
an import of scipy ever succeeds, the environment has drifted from the one of
record.

---

## 3. Obtaining XFOIL

XFOIL is not in this repository and must not be. It is obtained separately.

1. Download the XFOIL 6.99 Windows distribution from its own source. The build of
   record used the directory `XFOIL6.99/`, holding `xfoil.exe` as a Win32
   executable, alongside `Xfoil699src.zip` and `plotlib/`.

2. Note the licence before you redistribute anything. XFOIL 6.99 is by Mark
   Drela and is licensed under the GNU General Public License, version 2 or, at
   the user's option, any later version. The copyright notice reads
   Copyright (C) 2000 Mark Drela. This repository does not ship the binary, and
   each user obtains it under its own terms.

3. **Put `XFOIL6.99/` in the project root.** The code fixes that path:
   `evaluate.py` sets `XFOIL_BINARY = Path("XFOIL6.99") / "xfoil.exe"`, and
   `evaluate.solver_settings` passes it to `solver.run_polar`, which takes the
   binary as an argument and locates nothing itself. To put the distribution
   anywhere else, change that one constant.

Nothing in this build calls the plot library.

---

## 4. Environment setup

1. Create a virtual environment. The build of record used `.venv/` in the project
   root.

```bash
python -m venv .venv
```

2. Activate it. On Windows PowerShell, run the activation script.

```bash
.venv/Scripts/Activate.ps1
```

3. Install torch first. The `torch==2.13.0+cpu` local version does not resolve
   from the default package index. The article records the version string and
   does not record the index or the install command used, so you must supply
   that yourself. This is the one gap in the install that the repository cannot
   close for you.

4. Install the rest.

```bash
pip install -r requirements.txt
```

5. Copy the published artifacts to the project root, where the code reads them.

```bash
cp artifacts/*.npz artifacts/*.pt artifacts/*.jsonl .
```

---

## 5. Verifying the install

Four checks, in order. Run all four before any build step.

### 5.1 Check the interpreter and the libraries

```bash
python --version
```

Expect `Python 3.14.6`.

```bash
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

Expect `2.5.1 2.13.0+cpu False`. A `True` on the last value means a CUDA build is
installed and the environment does not match the one of record.

### 5.2 Check the artifacts arrived intact

```bash
cd artifacts && sha256sum -c SHA256SUMS && cd ..
```

Ten lines, all `OK`. A standing diagnostic inside the loader exists to catch a
stale artifact silently rescaling everything downstream, and it will raise at the
first driver you run, but the checksums are the cheaper check and cost nothing.

### 5.3 Let the standing load diagnostics run

Six standing diagnostics, D01 to D06, run on every execution and cannot be
skipped. Five of them live in `model.load_build_artifacts`, which is the single
loader every driver reads. The sixth lives in `solver.run_polar`. They read and
they raise. They compute nothing any caller uses and they change no value. Their
tolerances are in `model.STANDING_LOAD_DIAGNOSTICS`.

You do not run them directly. The first driver you launch loads the artifacts and
they fire. What they catch is the following.

| Diagnostic | Catches |
|---|---|
| D01 | A drifting or mis-applied standardisation scale, and a broken label round trip |
| D02 | A stale artifact silently rescaling everything downstream |
| D03 | A split that overlaps, loses rows, or collapses its stratification |
| D04 | An inverted flag, which would read the two arms backwards while every shape and count still looked correct |
| D05 | Ensemble reseeding failing, which would make the members identical and the spread penalty a constant zero |
| D06 | A timer measuring the wrong span, which would leave the committed timeout unjustified |

If any of these raises on a clean install, the files you copied at step 5 of
section 4 do not match the code. Fix that before going further.

### 5.4 Confirm the solver responds

The pipeline's own solver check is gate zero, inside step B19 at section 6.4. It
solves one known airfoil, `seeds/e387.dat`, through the same path at the
committed operating point and timeout, reads the coordinates raw without
round-tripping them through the fit, and requires a converged status on all 9
requested angles. A responding binary, a partial sweep or a written-but-empty
polar all fail.

On the build of record it converged on 9 of 9 in **0.728 seconds**, which section
2.5 of the article records.

There is no standalone command for gate zero. Run B19 in its normal position and
read its result there.

---

## 6. The build steps, in run order

The run order comes from the B step identifiers. The retained drivers are B16,
B17, B18, B19, B20, B21, B23, B24 and B25.

**There is a gap at B22.** The identifier appears nowhere in the article and no
driver carries it. The numbered drivers step straight from B21 to B23. Whether
B22 was a step whose driver was deleted, or was never a step at all, is not
stated. Nothing here fills it in.

**Steps B01 to B15 have no retained driver.** Their outputs are the artifacts you
copied in section 4. There is no command for these steps and this manual writes
none. Section 2.1 says what that does and does not mean.

Run each command from the project root, with the virtual environment active.

Only `run_b23_paired_evaluation.py` takes command line arguments. For the rest
the bare invocation below is the assumption; a driver's own source is the
authority.

Each driver's own docstring states what it does, its position in the run order,
what it reads and what it writes. This section is the same information in one
place.

### 6.1 B16, the prior weight sensitivity sweep

```bash
python run_b16_weight_sweep.py
```

| | |
|---|---|
| Reads | the build artifacts, through `model.load_build_artifacts` |
| Writes | `sweep/`, being `sweep_table.json` plus one history file and one checkpoint per candidate |
| Runtime | 21 distinct training runs in 309 s, per Table A3. A training cost, not a solver cost |

Sweeps each of three weights one at a time with the other two at zero, and
records the validation reconstruction, the mean distance to the reference per
arm, the diversity statistic and the live latent dimension count per candidate.
Raises rather than proceeding if a term's value at initialisation is exactly
zero, because its ladder scale would be undefined.

The diversity figures here sit on the sweep's internal grid of 11 evenly spaced
normalised targets from 0.0 to 1.0. That is not the requested target band used
later. Table 6 of the article says the same, and figures on the two grids are not
comparable point for point.

### 6.2 B17, weight selection

```bash
python run_b17_select_weights.py
```

| | |
|---|---|
| Reads | `sweep/sweep_table.json`, and nothing else |
| Writes | `b17_selection.json` |
| Runtime | not recorded in the article |

Applies the committed rule, being the highest score of effect gain less
reconstruction cost less diversity cost on min-max normalised components. The
rule carries no tolerance constant by design. An exact tie goes to the smaller
weight. On the build of record the selection was a single maximum with no tie, at
a prior weight of 5.862918756788936, which Table A1 records.

### 6.3 B18, training and the prior mechanism gate

```bash
python run_b18_prior_mechanism_gate.py
```

| | |
|---|---|
| Reads | `b17_selection.json`, and the build artifacts |
| Writes | `committed_model.pt`, `control_model.pt`, `b18_gate.json` |
| Runtime | 2 training runs at 150 epochs each, per Table A3. No wall clock is recorded |

Trains the committed conditional VAE at B17's weights, then runs the three group
verification suite Table 7 reports.

- Group one, arm blindness. Every term except the prior must be bit-identical
  across a flag flip, at a tolerance of exactly zero.
- Group two, gate response. The prior term must be exactly zero with flags clear
  and strictly greater than zero with flags set.
- Group three, the effect. Direction consistency at least 0.60, mean distance
  separation at least 0.1300377028358953, and arm effect over sampling noise at
  least 0.25. Table A2 carries all three.

A zero-weight control must pass groups one and two and fail every test in group
three, against these same unchanged thresholds. The gate generation grid is 11
targets by 20 samples on the internal normalised grid.

### 6.4 B19, the evaluation gates

```bash
python run_b19_evaluation_gates.py
```

| | |
|---|---|
| Reads | `dataset.npz`, `population.npz`, the build artifacts, plus the XFOIL binary |
| Writes | `b19_gates.json` |
| Runtime | gate zero, 1 solver call at 0.728 s. No total is recorded for gate one's 25 calls |

Gate zero is the solver responsiveness check described at section 5.4.

Gate one is the pipeline consistency check. Twenty-five dataset rows whose labels
are already known are pushed back through the same decode and solver path the
evaluation uses, and the recomputed labels are compared against the stored ones.
Five rows come from each of the five families, at the 0th, 25th, 50th, 75th and
100th percentile of that family's own stored label. The selection is
deterministic and draws no random number. Every one of the 25 rows must satisfy a
relative difference at or below 0.01, which Table A2 records. One row outside it
fails the gate and the failing rows are named.

On the build of record the maximum relative difference was 0.0 across all 25
rows.

Both gates must pass before any generated shape is evaluated.

### 6.5 B20, the truncation analysis

```bash
python run_b20_truncation_analysis.py
```

| | |
|---|---|
| Reads | `b07_labelling_progress.jsonl` |
| Writes | `b20_truncation.json` |
| Runtime | not recorded in the article |

Measures how much a maximum taken from the first k converged points understates
the maximum over the whole sweep, across the 862 fully converged sweeps in the
dataset. Selects the smallest k for which the mean absolute relative bias is at
most 0.010 and the 95th percentile of the absolute relative bias is at most
0.020. Table A2 carries both tolerances.

**Note the input.** This step reads B07's own stored sweep records, not
`dataset.npz`. It needs every converged point of every sweep, and `dataset.npz`
stores only the label each sweep produced. That file is in the published archive
for this reason.

On the build of record it selected 8 of 9, which Table A1 records. Seven
satisfies the mean clause at 0.0053 and fails the tail clause at 0.0442. Eight
satisfies both at 0.0018 and 0.0133. Section 2.5 of the article gives the same
four figures.

### 6.6 B21, the paired yield and the launch target

```bash
python run_b21_paired_yield.py
```

| | |
|---|---|
| Reads | the build artifacts, `committed_model.pt`, plus the XFOIL binary |
| Writes | `b21_paired_yield.json` |
| Runtime | not recorded in the article. Section 4.4 gives only an aggregate for the whole build |

Runs a pilot and measures the fraction of launched pairs in which both members
clear admission. Divides the committed floor of 100 analysed pairs by that yield
to derive the launch target. Raises if the measured yield is at or below zero,
because the launch target would then be undefined.

On the build of record the yield was 0.990909 and the launch target 101 pairs,
realised as 11 targets by 10 samples, which is 110 pairs. Table A1 records the
floor, the launch target and the yield.

### 6.7 B22

No driver exists and the article never mentions the identifier. Skip it.

### 6.8 B23, paired generation and evaluation

```bash
python run_b23_paired_evaluation.py
```

| | |
|---|---|
| Reads | `committed_model.pt`, `b18_gate.json`, `b21_paired_yield.json`, the build artifacts, plus the XFOIL binary |
| Writes | `b23_evaluation.json`, `b23_evaluation_progress.jsonl` |
| Runtime | 220 solver calls totalling 265.3 s, mean 1.206 s, maximum 6.912 s, per Table A3. The step also retrains the zero-weight control, and no wall clock is recorded for the step as a whole |

Retrains the zero-weight control from B18's recorded weight vector and training
seed, then generates matched pairs. The two arms share one trained model, one set
of weights, one conditioning layout and one latent code per pair. They separate
at a single term in the training objective and nowhere else.

Every generated shape is solved and lands in exactly one status, so kept plus
every discard category equals attempted. A record is admitted when a label is
present and at least 8 of the 9 sweep points are usable. The floor test requires
at least 100 pairs with both members admitted. Table 8 reports what this run
produced.

A resume guard refuses to continue a rerun whose stored header disagrees on the
seed, the grid, the model, the sample count or the admission rule.

### 6.9 B24, the analysis

```bash
python run_b24_analysis.py
```

| | |
|---|---|
| Reads | `b23_evaluation.json`, `normalization.npz` |
| Writes | `b24_analysis.json`, and `RESULTS.txt` |
| Runtime | not recorded in the article |

Computes the primary outcome, being the mean paired difference in target
satisfaction error, with a wild cluster bootstrap-t interval at 9999 resamples
and the unrefined percentile cluster bootstrap reported beside it. Also computes
the endpoint Monte Carlo error over 20 repetitions per estimator, the paired
difference distribution shape, the four secondaries, the trimmed mean sensitivity
check, the admission exclusion count, the differential attrition test and the two
per-arm mean converged point counts.

This step is the sole source of every reported statistic and every reported count
in the article, together with B25. Counts are recomputed here from the evaluation
records rather than carried over from the generation run.

The interval is read in code and not in the writing, through
`analysis.interval_reading`. A lower bound at or below zero with an upper bound
at or above zero reads as no detected difference.

### 6.10 B25, the metrics

```bash
python run_b25_metrics.py
```

| | |
|---|---|
| Reads | `b23_evaluation.json`, `b24_analysis.json`, the build artifacts, `committed_model.pt`, plus the XFOIL binary |
| Writes | `b25_metrics.json`, `b25_condition_blind_progress.jsonl`, and its section appended to `RESULTS.txt` |
| Runtime | 220 solver calls totalling 252.0 s, mean 1.146 s, maximum 5.861 s, for the condition-blind baseline, per Table A3. No wall clock is recorded for the step as a whole |

Computes the per-arm target tracking slopes and correlations, the condition-blind
baseline, the generative diversity metric on 11 targets by 20 samples, the
per-arm evaluability rates and the surrogate to solver gap. Tables 12, 13 and 14
report what this run produced.

Two standing checks run first and both can stop the step. B23's paired geometry
is regenerated from B23's own generation seed and must equal the stored
coefficients at a worst deviation of exactly 0.0. The surrogate held-out error
read from the parameter record must match that record's own derivation text.

### 6.11 B26

The appendix assembly step. It has no driver and produces no artifact.

---

## 7. Figures

Two drivers sit outside the numbered build steps. Run them in this order, after
B25.

### 7.1 Regenerate the training history

```bash
python run_committed_training_history.py
```

| | |
|---|---|
| Reads | `b17_selection.json`, `b18_gate.json`, `committed_model.pt`, the build artifacts |
| Writes | `committed_training_history.json` |
| Runtime | not recorded in the article. It reproduces one 150 epoch training run |

No numbered step stored the per-epoch objective components or the per-dimension
divergence that Figures A1 and A2 plot, so this driver regenerates B18's
committed training run with them logged.

It exits without writing unless four things hold. The weights read must equal
B18's recorded weights, the training seed must equal the offset rule's value, the
live dimension count must match, and the reproduced best checkpoint must equal
`committed_model.pt` tensor by tensor.

### 7.2 Draw the figures

```bash
python run_figures.py
```

| | |
|---|---|
| Reads | `b18_gate.json`, `b20_truncation.json`, `b23_evaluation.json`, `b24_analysis.json`, `b25_metrics.json`, `b17_selection.json`, `committed_training_history.json`, `committed_model.pt`, `avian_signature.npz`, `sweep/sweep_table.json`, the build artifacts |
| Writes | `figures/`, ten figures as one PDF and one 200 dpi PNG each |
| Runtime | not recorded in the article |

The ten figures are named for the article's own numbering, and the article prints
each one's path in its caption.

| Article | File |
|---|---|
| Figure 1 | `figures/figure_1_method_schematic.pdf` |
| Figure 2 | `figures/figure_2_prior_weight_sensitivity.pdf` |
| Figure 3 | `figures/figure_3_distance_to_reference.pdf` |
| Figure 4 | `figures/figure_4_paired_difference.pdf` |
| Figure 5 | `figures/figure_5_signed_error_per_arm.pdf` |
| Figure 6 | `figures/figure_6_shape_family_against_reference.pdf` |
| Figure 7 | `figures/figure_7_diversity_across_range.pdf` |
| Figure 8 | `figures/figure_8_surrogate_against_solver.pdf` |
| Figure A1 | `figures/figure_A1_training_curves.pdf` |
| Figure A2 | `figures/figure_A2_latent_dimension_usage.pdf` |

Each also writes a 200 dpi PNG under the same stem. `figures/` is committed to
this repository, so the paths the article prints resolve in a fresh clone without
a rerun.

Two further figures are built and not written. The attrition flow and the
truncation bias are no longer cited by the article, so no image is produced for
either. Their code is retained and both still run, because their consistency
checks are part of this driver's seven. Nothing is lost: Table 8 carries the
attrition and section 2.5 carries the truncation rule.

This driver carries seven checks of its own and any of them can stop it.
Figure 5's fitted slope must match the recorded tracking slope less one to within
1e-12. Figure 7's recomputed across-range mean must match the recorded mean to
within 1e-12. Figure 2's committed weight must appear in the sweep's own ladder.
Figure 8's three recomputed scalars must equal B25's recorded values exactly.
Figure 3 must reproduce `b18_gate.json` at exact float equality for both arms'
mean distance and for direction consistency, and must match
`avian_signature.npz` at exact integer equality on both separation counts.
Figure A2's live dimension count must match the regenerated count, the stored
history and `b18_gate.json` exactly. The truncation bias figure's bias at the full
sweep length must be exactly 0.0 on both the mean and the upper percentile.

Text on the figures is limited to titles, axis labels, legends and labels
attached to a mark. The explanation of each figure is the article's caption and
is not repeated inside the image.

matplotlib is used here and by no pipeline module.

---

## 8. The optional falsification check

```bash
python check_b24_studentised_secondaries.py
```

| | |
|---|---|
| Reads | nothing from the build. It runs its own simulation |
| Writes | nothing. The verdict goes to the console |
| Runtime | not recorded. The design is 11 clusters by 10 pairs, 400 datasets per design, two designs, 399 resamples |

**This check is retained because it failed, and rerunning it will fail again.** It
returned FAIL on the design B width agreement clause, `C_AGREE_TOLERANCE = 0.40`,
for both statistics tested. With no between-cluster component the refined interval
runs 2.82 times wider than the pair interval for the median and 1.60 times wider
for the slope, against that tolerance, which was fixed in advance and appears in
Table A2. No threshold was moved and no clause was deleted.

Section 3.4 of the article reports the failure and section 4.6 repeats it. Read
`RESULTS.txt` section 4 before quoting any of the four secondaries.

---

## 9. Where the reported numbers land

Every reported statistic and every reported count comes from `analysis.py`, wired
by B24 and B25. `RESULTS.txt` is their captured output. A number that does not
appear in it has no source and does not appear in the article.

| Where to look | What it carries |
|---|---|
| `PAPER.md` | Every reported number, at full precision, in Tables 2 and 4 and 6 to 14, and in Tables A1 to A3. This is the fastest place to check a value against a rerun |
| `RESULTS.txt` | The same numbers as captured output. Section 4 carries the consequence of the failed falsification check for reading the four secondaries |
| `b24_analysis.json` | The primary outcome, both intervals, the Monte Carlo error, the distribution shape, the four secondaries, the sensitivity check and the recomputed counts |
| `b25_metrics.json` | The tracking slopes and correlation, the condition-blind baseline, the diversity metric, the evaluability rates and the surrogate to solver gap |
| `b18_gate.json` | The mechanism quantities section 3.1 and Table 7 report, being M12, M13, M14 and the gate verdicts. These are gate outputs measured before evaluation |
| `b20_truncation.json` | The truncation bias table and the selected minimum converged point count |
| `b23_evaluation.json` | The attrition flow and the per-record evaluation results, which B24 recounts rather than trusts |
| `artifacts/` | Ten numbers reported in section 2.2.2, section 2.5, Table 3, Table 4 and Table A1, readable without running anything. `artifacts/README.md` lists which file carries which |
| `figures/` | Figures 1 to 8, A1 and A2 |

`RESULTS.txt` and the step records are regenerable and are excluded through
`.gitignore`. `figures/` is committed, because the article prints each figure's
path. `artifacts/` is committed, because nothing in the repository can rebuild it.

No number in the article's appendix is a result. The appendix carries the
pre-registration record, the committed parameters, the thresholds and the
environment.

No number is taken from a diagnostic printed during a generation run.

---

## 10. Total wall clock

The article records solver call cost and one training total. It does not record a
wall clock total for any driver, and it does not record one for the build as a
whole. So no total expected wall clock is stated here, because none exists to
state.

What Table A3 records is the following, together with gate zero from section 2.5.

| Run | Recorded cost |
|---|---|
| Dataset labelling pass, B07 | 1005 solver calls, 1548.3 s total, 1.541 s mean, 7.597 s maximum |
| Paired generation and evaluation, B23 | 220 solver calls, 265.3 s total, 1.206 s mean, 6.912 s maximum |
| Condition-blind baseline, B25 | 220 solver calls, 252.0 s total, 1.146 s mean, 5.861 s maximum |
| Weight sensitivity sweep, B16 | 21 training runs, 309 s |
| Committed model and control, B18 | 2 training runs, 150 epochs each. No wall clock |
| Gate zero, B19 | 1 solver call, 0.728 s |

Adding every figure the article records gives about 2375 s, being about 40
minutes. That sum is a floor on the whole build and not an estimate of it, because
it omits every unrecorded step and every non-solver cost inside the recorded ones.

Across the retained drivers alone, being B16 through B25, the recorded cost sums
to about 827 s, being about 14 minutes. The B07 labelling pass is the largest
recorded item in the build and has no retained driver, so it does not appear in a
rerun from this repository.

The article gives one aggregate for the build as a whole, at section 4.4: roughly
2,700 solver calls plus 24 training runs. Per-step call counts for the
instrumentation pass and the B21 pilot are not recorded, and none is stated here.

**What dominates.** The solver dominates the recorded cost. Within the retained
range, the B16 sweep at 309 s and the two 220 call solver runs at 265.3 s and
252.0 s are the three largest recorded items. B24 does four 9999 resample
bootstraps, an exact 2048 vector enumeration and 20 Monte Carlo repetitions per
estimator, and its cost is not recorded anywhere.

---

## 11. Determinism

### 11.1 The seed derivation rule

One base seed, 20260806. Every stream is the base seed plus 1000 times the build
step number plus a substream index. Both live in `dataset.rng_for`, and the base
seed is carried inside `flag_assignment.npz` and `split.npz` as well.

The article states at Appendix A.3 that every source of randomness derives from
one documented base seed by a stated offset rule. It does not print the seed or
the formula; the repository does, at the two places above.

Every downstream generator derives from that one seed by that rule, and every
distinct source of randomness is recorded.

### 11.2 The determinism observation, at its exact scope

The analysis module was re-run once after the build was otherwise complete, to
add two per-arm means to its output. The re-run produced a file differing from the
previous one in exactly two respects, being the recorded run timestamp and the new
block. Every other line was byte-identical. That covers all four 9999 resample
intervals on the primary and the secondaries, the exact 2048 vector enumeration,
the 40 independent Monte Carlo repetitions behind the endpoint error metric, every
secondary and sensitivity statistic, and every recomputed count.

**What it supports.** That the seed derivation rule does what it claims, so a
given step's generators are reconstructible from the one base seed and the offset
rule alone, and that no reported statistic in the build depends on an unrecorded
source of randomness.

**What it does not support.** One module, re-run once, on the same machine, in the
same virtual environment, at the same library versions, reading the same stored
evaluation records. It is a determinism observation and not a reproduction. It
says nothing about a different machine, a different library version, a rebuilt
dataset, or a retrained model, and none of those was attempted.

Do not describe this pipeline as reproducible beyond that. The article makes no
clean-room reproduction claim, no cross-machine claim and no portability claim of
any kind.

### 11.3 One value that is not recoverable

The sampler's per-seed loop guard, `max_trials_per_seed`, was passed by a driver
that was deleted, and its committed value appears in no artifact and in no record.
No number is asserted for it anywhere.

What is established is that it does not matter. B06's draw was replayed at the
recorded base seed on substream 1 and reproduced the committed population bit for
bit across all 1005 rows and both coefficient blocks. The trials actually taken
were 200 for e387, 200 for s1223, 201 for sd7003, 200 for seagull and 200 for
sg6043. The guard can only bind at 201 or below, so the committed population is
invariant at every value of 202 or more, and the committed run completing proves
the value used was at least 202.

---

## 12. Troubleshooting

### 12.1 The solver hangs and never returns

Sweeping a real seed into separation-prone angles can make XFOIL hang
indefinitely with no distinguishing output before the hang, and the hang is
sequence dependent within one accumulated session rather than a fixed property of
any one angle. The per-call timeout of 7.56 s exists for this, and Table A1
records it. The wrapper kills and reaps the subprocess explicitly, with a reap
timeout of 10 s after the kill. On this platform the default call does not
reliably terminate the child.

If orphaned XFOIL processes accumulate across a long run, the kill and reap path
is not working. It is not optional plumbing.

### 12.2 A shape is recorded as a timeout rather than a convergence failure

That is intended. Timeouts are reported separately from true convergence
failures, always. A timeout means the shape is slow and not that it is
unsolvable.

On the dataset of record every one of the 23 discards was a timeout. Not one shape
failed to converge for an aerodynamic reason and not one produced an environment
fault. Table 2 gives the breakdown: the timeouts fall unevenly across families,
with E387 losing 13 of the 23 and SG6043 losing none.

The timeout itself is derived, being the 95th percentile of measured successful
solve times over a full population-scale pass times a margin of 2.0, which Table
A1 records.

### 12.3 Every solve fails and the polar is empty

Suspect the staging path first. The solver stores filenames in a fixed-width
buffer. A long path truncates and the load fails silently, producing no polar and
no error a caller could distinguish from an aerodynamic failure. The wrapper
copies each airfoil to a short relative filename inside a staging directory and
runs the solver with its working directory set there.

The staging directory is `_stage/`, at `evaluate.STAGE_DIR`, with `_scratch/` at
`evaluate.SCRATCH_DIR` beside it. If you have moved the project to a deep path,
or changed either constant, this is the first thing to check.

### 12.4 A run is classified as an environment fault

The wrapper raises this when no polar accumulation header is written, or when
`LOAD NOT COMPLETED` appears in the console output. It means the solver did not
get as far as solving. Check the binary, the path and the staging directory
before looking at the airfoil.

A run that completes the whole command stream and genuinely converges none of the
requested angles is a different thing: it still writes a polar file with a header
and no data rows.

### 12.5 A driver refuses to resume

B23 carries a resume guard. A rerun whose header disagrees with the stored header
on the seed, the grid, the model, the sample count or the admission rule refuses
to resume. Do not edit the stored header. Delete `b23_evaluation.json` and start
that step again if the disagreement is intended.

### 12.6 A driver exits on a regeneration check

Three guards behave this way and each names what it compared.

- B25 stops when B23's paired geometry regenerated from B23's own generation seed
  does not equal the stored coefficients at a worst deviation of exactly 0.0.
- B25 stops when the surrogate held-out error read from the parameter record does
  not match that record's own derivation text.
- `run_committed_training_history.py` exits when the weights, the training seed,
  the live dimension count or the reproduced best checkpoint do not match B18's
  committed record.

Each of these means an artifact and the code that reads it have drifted apart.
Rebuild the artifact rather than loosening the check.

### 12.7 A driver raises on a guard rather than running

- B16 raises when a term's value at initialisation is exactly 0.0, because its
  ladder scale is undefined.
- B21 raises when the measured pair yield is at or below 0.0, because the launch
  target would be undefined.
- The standardisation step raises when any column's standard deviation is exactly
  0.0.
- `geometry.read_seed_dat` rejects a seed file whose abscissae are not monotone
  within a tolerance of -1e-9 on the successive difference.

### 12.8 A standing diagnostic raises at load

D01 to D06 fire inside the loader and inside the solver wrapper. Each was verified
to fire on an artifact doctored in exactly the way it exists to catch, so a raise
is a real finding and not a false alarm. Section 5.3 lists what each one catches.

D02 in particular catches a stale artifact silently rescaling everything
downstream. If you did not copy the artifacts from `artifacts/` in this
repository, run the checksums at section 5.2 first.

### 12.9 An import fails on scipy

Nothing in this build imports scipy and none is installed. `requirements.txt`
says so and says why. If something asks for it, the environment has drifted from
the one of record. Do not install it.

---

## 13. Attribution for the seed files

The five files under `seeds/` carry no licence statement in this build's inputs,
and Table A3 asserts none. What follows is source attribution and it is not a
licence grant. `seeds/SOURCES.md` carries the same attribution beside the files.

Four of the five sections come from the UIUC Airfoil Coordinates Database,
maintained by Michael Selig at the University of Illinois at Urbana-Champaign.
The database holds approximately 1,650 airfoils at version 2.0 and its archive
was last updated 23 February 2026.
https://m-selig.ae.illinois.edu/ads/coord_database.html

- `seeds/e387.dat`, Eppler E387
- `seeds/s1223.dat`, Selig S1223
- `seeds/sd7003.dat`, SD7003
- `seeds/sg6043.dat`, SG6043

The fifth was constructed from the seagull wing cross section reported in Liu, T.,
Kuykendoll, K., Rhew, R., and Jones, S. (2006). Avian wing geometry and
kinematics. AIAA Journal, 44(5), 954 to 963. https://doi.org/10.2514/1.16224

- `seeds/seagull.dat`

Anyone needing permission to reuse or redistribute the four database files should
seek it from the database. Anyone needing permission for the avian section should
seek it from the article and its publisher. See `seeds/SOURCES.md`, which
carries the same attribution beside the files themselves.
