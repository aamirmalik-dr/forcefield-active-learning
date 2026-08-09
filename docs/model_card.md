# Model card: committee_active.pt

## What this is

A deep-ensemble committee of five compact interatomic potentials for BCC
Ti-Zr-Nb solid solutions, produced by the query-by-committee active-learning
loop of this repository (committee strategy, seed 0, full 300-label budget).
The committee mean is the deployed potential; the committee spread is its
uncertainty signal.

- Architecture per member: compact atom-centered symmetry functions (8
  element-resolved radial G2 x 3 neighbor elements + 6 element-agnostic
  angular G4 = 30 features, cutoffs 5.5 / 4.5 A), per-element SiLU MLPs
  (16, 16), float64, 2,355 parameters per member, 11,775 for the
  committee. Forces are exact autograd derivatives.
- Energies are a composition-linear reference (least-squares per-element
  reference energies, fit on purchased frames only) plus the learned
  residual.
- Members differ only in initialization seed and batch order; each trains
  from scratch on all purchased frames every acquisition round (Adam,
  cosine decay, batch 16, force weight 0.1, no early stopping, and the
  equal-effort epoch rule min(120, max(40, 12000/n)) so small purchased
  sets are not starved of optimizer steps).

## Training data, plainly

Labels are SURROGATE MODEL LABELS from chgnet 0.4.2 (CHGNet v0.3.0
checkpoint, BSD-3-Clause), chosen over MACE-MP-0 small by measured CPU
speed (data/teacher_choice.json). They are not DFT, and no number in this
repository is DFT accuracy. The committee saw exactly the 300 frames its
loop purchased from a 16-to-54-atom BCC rattle/strain pool (7 compositions;
Ti2ZrNb and TiZrNb2 held out entirely).

## Measured performance (fresh venv, this session)

Held-out-generation-group test set, committee mean, seed 0 at budget 300;
the benchmark's 3-seed mean and spread are in RESULTS.md and
results/metrics.json:

| Metric | Value |
|---|---|
| Test energy MAE | 27.6 meV/atom |
| Test force MAE | 193.7 meV/A |
| Transfer (held-out compositions) force MAE | 250.7 meV/A |
| Calibration (Spearman rho, disagreement vs actual error) | 0.87 |
| NVE drift (300 / 900 K init, 2 fs, 3 ps, 16 atoms) | -0.0003 / -0.0001 meV/atom/ps |

Read these numbers with RESULTS.md in hand: the random-selection baseline
at the same budget is BETTER on this test set (175.1 meV/A, 18.1 meV/atom,
3-seed means), and the committee's equilibrium physics carries its
acquisition bias (see the EOS section). This checkpoint is committed as
the loop's artifact, not as a recommended production potential.

## Intended use and limits

- Intended: reproducing and extending this repository's benchmarks;
  committee-disagreement scoring of BCC Ti-Zr-Nb candidate structures;
  short MD on near-BCC configurations at moderate temperature.
- Not intended: chemistry outside Ti-Zr-Nb, non-BCC phases, surfaces,
  defects beyond the rattle/strain envelope, melts, or any production
  property prediction. Accuracy is bounded above by the teacher, and the
  teacher's own errors against DFT pass through unchanged.
- The committee's uncertainty is a rank signal, not a calibrated error bar:
  use it to order candidates, not as a confidence interval.

## Sibling checkpoints

- `models/committee_random.pt`: the same architecture at the identical
  300-label budget, fed by random selection (replayed exactly from the
  recorded purchases of the main benchmark's random seed-0 run by
  scripts/train_random_reference.py). Committed as the control for the
  acquisition-bias comparison in the EOS benchmark.
- `models/committee_md_explore.pt`: the same architecture trained by the
  MD-exploration variant (configs/md_explore.yaml, seed 0); committed so
  the exploration result can be inspected without a teacher install.
