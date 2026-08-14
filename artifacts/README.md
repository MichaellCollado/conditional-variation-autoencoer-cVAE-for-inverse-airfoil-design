# The pre-B16 archive

Ten files, 1,212,369 bytes. They are the inputs the retained drivers read and
cannot rebuild.

The retained drivers begin at build step B16. Every artifact they read is built
at a step below B16, and no retained driver builds any of them. Without this
directory a fresh clone cannot run anything. With it the repository is runnable
from B16 onward.

These are this build's own outputs, produced by this build's own code from the
five seed coordinate files under `seeds/`. Nothing here was obtained from
outside the study.

---

## How to use them

The code reads these files from the project root, not from here. `artifacts/` is
the published copy; the root is the working copy.

```bash
cp artifacts/* .
```

The root copies are excluded by `.gitignore` and the copies here are committed,
so the two never collide in version control. They are byte-identical as
published; the checksums below are how you confirm that after a download.

---

## What each file is

The B step numbers are this repository's own build vocabulary. They are how the
retained drivers are named, and they are not the article's. `PAPER.md` carries
no build step numbering, so nothing here is cited to it.

| File | Step | What it holds |
|---|---|---|
| `population.npz` | B06 | The drawn population before labelling. 1005 rows of upper and lower CST coefficients, the CST order, the perturbation width and the per-seed count |
| `dataset.npz` | B07 | The labelled dataset the model trains on. 982 rows, each a row index into the population, a family name, a seed flag and an XFOIL label |
| `normalization.npz` | B08 | Label normalisation. The label minimum and maximum, and the row set they were taken over |
| `standardization.npz` | B08 | Coefficient standardisation. Per-column mean and standard deviation over the training split alone, with the row set recorded in the file |
| `avian_signature.npz` | B09 | The reference the prior measures distance to. The raw and standardised avian signature, the percentile the region extent was set at, the extent itself, and the four separation counts |
| `flag_assignment.npz` | B11 | The flag draw. Which of the 982 rows are flag-clear, the requested and realised fractions, the balance statistics, and the seed the draw came from |
| `conditioning.npz` | not recorded | The conditioning array, 982 by 22, in the layout Table 3 gives. Written by `dataset.assemble_conditioning`. The step is not recorded anywhere; it runs at or after B11, because it consumes the flag draw |
| `split.npz` | B12 | The train and validation split. 785 training and 197 validation indices, the validation fraction and the seed the split came from |
| `surrogate_ensemble.pt` | B13 | The frozen surrogate. Five members, trained once and never retrained between arms |
| `b07_labelling_progress.jsonl` | B07 | B07's own per-sweep solver records, one JSON line per shape. Every converged point of every sweep, not just the label each sweep produced |

`flag_assignment.npz` and `split.npz` each record their own step in an
`rng_step` field, at 11 and 12. The other seven step numbers come from the build
order and are not carried inside the files.

## Which are needed to run, and which is here to explain

Nine of the ten are read at runtime. `model.load_build_artifacts` is the
single loader every driver goes through, and it opens `dataset.npz`,
`population.npz`, `conditioning.npz`, `split.npz`, `standardization.npz`,
`normalization.npz`, `avian_signature.npz` and `surrogate_ensemble.pt` on every
call. Remove any one and no driver starts.

`b07_labelling_progress.jsonl` is read by `run_b20_truncation_analysis.py`, and
by nothing else. B20 measures how much a maximum taken from the first k
converged points understates the maximum over the whole sweep, so it needs
every converged point of every sweep. `dataset.npz` stores only the label each
sweep produced, which is not enough, and no retained driver rebuilds these
records.

`flag_assignment.npz` is read by no code in this repository. It is here because
without it the archive cannot explain where the flag column in
`conditioning.npz` came from, and because it is the file that carries the
balance statistics the article reports.

## Checksums

SHA-256 for all ten files is in `SHA256SUMS`, beside them. Run it from inside
this directory.

```bash
sha256sum -c SHA256SUMS
```

A standing diagnostic inside `model.load_build_artifacts` exists to catch a
stale artifact silently rescaling everything downstream. If you obtained these
files from anywhere other than this repository, the checksums are the cheaper
check and you should run them first.

---

## Numbers you can check against the article without running anything

These files carry reported values directly, which makes them the shortest path
from a clone to a verified number.

| Article | Value | Where it sits in the archive |
|---|---|---|
| §2.2.2, Table A1 | Label range 50.49872773536895 to 172.17391304347828 | `dataset.npz`, `label` column, and `normalization.npz` |
| §2.2.2, Table A1 | 982 labelled rows of 1005 attempted | `dataset.npz` row count against `population.npz` row count |
| §2.2.2, Table A1 | Realised flag-clear fraction 0.095723, being 94 rows | `flag_assignment.npz`, `flag_clear_fraction_realised` and `flag_clear` |
| §2.2.2, Table A1 | 785 training and 197 validation rows | `split.npz`, `train_idx` and `val_idx` |
| §2.5 | Standardised mean difference -0.0357 | `flag_assignment.npz`, `cohens_d` |
| §2.5 | Distribution statistic 0.0653 against a critical value of 0.1473 | `flag_assignment.npz`, `ks_statistic` and `ks_critical_at_05` |
| §2.5 | All 10 well-populated deciles contain flag-clear rows | `flag_assignment.npz`, `n_well_populated_deciles` and `n_well_populated_deciles_missing_flag_clear` |
| Table 4 | Region extent 1.3003770283589529 | `avian_signature.npz`, `extent` |
| Table 4 | 175 of 195 avian rows and 0 of 787 non-avian rows inside the extent | `avian_signature.npz`, the four `n_*_inside` and `n_*_total` fields |
| Table 3 | Conditioning layout, column 0 target, columns 1 to 20 signature, column 21 flag | `conditioning.npz`, `array` and `n_signature_columns` |

The chain between the files is internally consistent and was checked rather than
assumed. The flag column of `conditioning.npz` reproduces `flag_clear` from
`flag_assignment.npz` exactly; every flag-set row carries the standardised
signature from `avian_signature.npz` and every flag-clear row carries a zero
block.

---

## What this archive does not do

It does not reproduce the steps that built it. The drivers for B01 to B15 were
deleted from the build after their falsification checks passed and are not in
this repository.

The modules that do the work of those steps are all present, which is what
`PAPER.md` claims under Code availability. `geometry.py` fits the seeds,
`dataset.py` draws the population, assembles the conditioning and splits the
data, `prior.py` derives the region extent, `model.py` trains and freezes the
surrogate. What is missing is the driver scripts that called them in order and
wrote these files out.

So the honest statement is that the code which produced every file here is in
the repository, and the scripts that ran it are not. Re-deriving these ten
files from `seeds/` alone would mean rewriting those drivers from the parameter
record in `params.py`.

## On `surrogate_ensemble.pt` and the "no pretrained model" claim

Code availability in `PAPER.md` says "No pretrained model is included, since both
models are trained from the labeled dataset by the code in the repository."
`surrogate_ensemble.pt` is a trained model and it is in this directory, so the
two sentences deserve reconciling rather than leaving to a reader to notice.

Nothing here is pretrained in the sense that phrase carries: no weights were
obtained from outside this study, no checkpoint was adopted from another model or
another dataset, and nothing was fine-tuned from anyone else's starting point.
The surrogate ensemble is this build's own output, trained at B13 from the 982
labelled rows by `model.build_surrogate_ensemble`, which is in the repository.

What is true, and is the qualification worth stating, is that B13's driver was
deleted with the rest of the pre-B16 drivers. So the trainer is here and the
script that ran it is not, and the frozen ensemble is published rather than
retrained on a fresh clone. The conditional VAE, which is the model the study is
about, is trained from the labelled dataset by `run_b18_prior_mechanism_gate.py`
on every run, and is not shipped.
