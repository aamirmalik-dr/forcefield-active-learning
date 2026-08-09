# Results

Every number below was measured this session in a fresh CPU venv; the
benchmark JSONs in results/ are the source of record and
results/metrics.json aggregates them. Labels are surrogate model labels
from chgnet 0.4.2 (CHGNet v0.3.0 checkpoint), never DFT. Errors are meV/atom
for energies and meV/A for forces, committee-mean predictions, held-out
generation groups unless stated otherwise.

## Setup

- Pool: 1,056 candidate frames (66 generation groups, 7 compositions,
  16 to 54 atoms; rattle amplitudes 0.04 to 0.28 A plus strain batches).
- Test: 288 frames from held-out generation groups of the same
  compositions; transfer: 144 frames of two fully held-out compositions
  (Ti2ZrNb, TiZrNb2).
- Committee: 5 members, 2,355 parameters each (30 compact ACSF features,
  per-element (16, 16) SiLU MLPs, float64, autograd forces).
- Budget schedule: 60 random initial labels (identical draw for all
  strategies per seed), then 6 rounds of 40. Training is retrain-from-
  scratch each round with an equal-effort epoch rule
  (min(120, max(40, 12000/n)) epochs on n frames), Adam, cosine decay,
  force weight 0.1, no early stopping, identical for every strategy.
- Three seeds per strategy; seeds fix the initial draw, member inits, and
  random purchases.

## Main benchmark: the committee does not beat random here

Final errors at the 300-label budget (mean +/- population std over 3 seeds,
results/main.json):

| Strategy | Test F MAE (meV/A) | Test E MAE (meV/atom) | Transfer F MAE | Transfer E MAE |
|---|---|---|---|---|
| Committee, pure top-k | 185.6 +/- 5.9 | 28.11 +/- 0.39 | 243.9 | 42.2 |
| Committee, group cap 4 | 183.5 +/- 0.7 | 28.44 +/- 0.31 | 245.2 | 46.3 |
| Random selection | **175.1 +/- 6.0** | **18.07 +/- 0.37** | **238.2** | **29.4** |

Random selection wins at essentially every budget beyond 100 labels, on
both metrics, in and out of distribution. The label-reduction factor this
project set out to measure is **1.0 (not reached)**: neither committee
variant ever gets down to random's final error within the budget
(results/metrics.json, label_reduction). The energy gap is the striking
one: a factor of 1.56 in favor of buying labels blindly.

![error versus labels and acquisition map](figures/hero.png)

## Why: the acquisition map and the per-kind breakdown

The acquisition map (hero, right panel) shows what committee disagreement
buys: rounds 1 and 2 are almost entirely rattle-0.28 frames, and later
rounds walk down the amplitude ladder only as the hardest kinds run out.
The pool is roughly uniform over kinds; the purchases are not.

The per-kind breakdown shows the committee got exactly what it paid for,
and it still was not enough:

![final error by batch kind](figures/error_by_kind.png)

- On the kinds it bought (rattle 0.28, rattle 0.20) the committee beats
  random, e.g. about 1,270 versus 1,430 meV/A RMSE on rattle-0.28 test
  frames.
- On every other kind, random wins, and the energy calibration of the
  committee models is visibly worse (28 vs 18 meV/atom overall), because
  near-equilibrium frames are underrepresented in what the committee buys.
- The group cap does not change the picture: the redundancy is at the
  batch-kind level, not the single-batch level, so capping picks per
  generation group only reshuffles which hard frames are bought.

The same skew shows up along the composition axis: chemically disordered
frames disagree more, so the committee buys the ternary at 25 percent
against a 15 percent pool share while starving the elementals (Ti gets 6
percent of purchases against 14, Zr 7 against 16). Remember that number
when reading the equation-of-state section: the active committee barely
bought elemental frames. The mean disagreement of what gets bought falls
about 6x from round 1 to round 6 as the pool's hard tail is consumed
(figures/acquisition_map.png).

## The capacity ceiling compresses everything

The identical committee trained once on the entire 1,056-frame pool
(configs/ceiling.yaml) reaches 141.0 meV/A and 13.1 meV/atom, the dotted
line on the hero curves. Two readings follow. First, the pool is not
exhausted at 300 labels: random still sits 34 meV/A above the ceiling, so
a better selector had room to win and did not. Second, the marginal value
of labels is shallow here: going from 300 to 1,056 labels buys only about
19 percent on forces, so even perfect selection could not have produced a
dramatic curve separation on this pool. Selection strategies matter most
where marginal value is steep; the scarce-pool benchmark below constructs
exactly that regime.

## The uncertainty signal itself is good

Query-by-committee did not fail because the uncertainty was wrong. On
held-out test frames, per-frame committee force disagreement ranks actual
force error with Spearman rho 0.87 to 0.91 across all six committee runs
(figures/calibration.png; results/metrics.json, calibration). The bin-mean
curve also shows the committee is systematically overconfident: actual
errors run about 2x the raw disagreement, so the signal is a ranking, not
an error bar.

![calibration](figures/calibration.png)

A well-calibrated ranking can still buy the wrong data: the frames with the
highest error are self-similar high-amplitude configurations whose labels
mostly teach the model about each other, while the average error over a
mixed test set is governed by data the committee refuses to buy.

## The scarce pool: the regime where the committee earns its keep

Same dataset, same test set, same training, but the pool now keeps every
easy group (rattle 0.04 and 0.08, 73 percent of 480 frames) and only two
generation groups of each hard kind (configs/scarce.yaml, 2 seeds,
results/scarce.json). Now a random buyer wastes most of its budget on
redundant near-equilibrium frames, and committee disagreement pays off:

![scarce-pool curves](figures/scarce.png)

- The committee reaches random's final-budget force error with about
  **2.2x fewer labels** (139 versus 300; energy 1.7x), the label-reduction
  factor the main pool never delivered.
- Final errors: committee 202.7 +/- 1.5 versus random 208.4 +/- 2.6 meV/A,
  and 26.1 versus 28.8 meV/atom. The endpoint margins are small; the
  label-efficiency gain along the curve is the result.
- Honest caveats: 2 seeds, not 3. Random is clearly AHEAD at 100 labels
  (205 vs 244 meV/A) and the committee's energy error spikes there
  (39 meV/atom), the same hard-frame-first failure as on the mixed pool,
  transient this time only because the hard kinds run out; the committee
  leads from 140 labels on. Its advantage consists of finding the rare
  hard groups quickly, which is precisely what its (well-calibrated)
  disagreement signal is good at.

Together the two pools bracket the practical answer: query-by-committee is
not a free lunch over a well-designed candidate pool, and it is worth
roughly a 2x label saving when informative candidates are rare, which is
the situation practitioners usually build active learning for.

## The split-protocol exhibit: the leaky test would have flipped the conclusion

Evaluating on sibling frames (unpurchased frames from the same generation
groups the models trained on) instead of held-out groups:

| Strategy | Honest F MAE | Sibling-frame F MAE |
|---|---|---|
| Committee, pure top-k | 185.6 | 114.8 |
| Committee, group cap 4 | 183.5 | 112.8 |
| Random | 175.1 | 134.3 |

The leaky protocol flatters everyone, but it flatters the committee more
(it trained on the hard groups the sibling set samples), so it does not
just shrink errors by roughly 40 percent, it **reverses the ranking**: a
random-frame evaluation would have reported committee acquisition beating
random by 16 percent, the opposite of the honest conclusion. This is the
one place this repository evaluates on leaky data, and it exists to make
that failure mode visible.

![split protocol](figures/split_protocol.png)

## Ensemble-MD exploration: mostly declined by its own committee

The exploration variant (configs/md_explore.yaml, 1 seed, stated as such)
lets the committee mean run short BAOAB Langevin trajectories each round
(1200 K, 7 compositions, from-scratch integrator) and injects the
snapshots as candidates; purchasing one costs a live teacher call through
the pool's oracle. The measured outcome is an honest shrug: across 240
acquisition purchases the committee bought its own MD snapshots only 8
times, preferring existing pool frames, and the final test error
(183.4 meV/A) sits inside the pure-committee seed spread (179.9 to 193.7).
On-policy exploration neither helped nor hurt here; the fixed pool already
contains harder frames than 1200 K committee-MD generates
(results/md_explore.json).

![md exploration](figures/md_explore.png)

## Physics checks: acquisition bias reaches the equation of state

Birch-Murnaghan fits of E(V) scans on ideal elemental BCC cells
(configs/eos.yaml), comparing the active committee against the
random-reference committee at the identical 300-label budget
(models/committee_random.pt, replayed exactly from the recorded purchases;
the replay reproduces the logged test error to 0.01 meV/A), the teacher,
and the real-DFT Materials Project anchors:

| | a0 (A) | B0 (GPa) |
|---|---|---|
| **Nb** active / random / teacher / DFT | 3.458 / 3.338 / 3.327 / 3.318 | 53 / 177 / 167 / 172 |
| **Ti** active / random / teacher / DFT | 3.257 / 3.237 / 3.268 / 3.252 | 78 / 81 / 74 / 111 |
| **Zr** active / random / teacher / DFT | 3.635 / 3.640 / 3.595 / 3.582 | 111 / 55 / 89 / 88 |

Nb is the clean verdict: the random-fed committee reproduces the teacher's
equation of state almost exactly (a0 within 0.011 A, B0 within 10 GPa),
while the active committee, which spent its budget on far-from-equilibrium
and mostly non-elemental frames (the acquisition map above), misses the
lattice constant by 0.13 A and collapses the bulk modulus to a third of
its value. Same architecture, same budget, same
training; only the purchase policy differs. Acquisition bias is not just a
test-set number, it is broken curvature where the potential was not fed.

![equation of state](figures/eos.png)

Honest footnotes: Ti B0 is wrong against DFT for every model including the
teacher (74 vs 111 GPa), the teacher's known miss, inherited. Zr is
weak for both students in opposite directions (111 and 55 vs the teacher's
89 GPa), consistent with Zr being the recipe's soft spot in the sibling
repositories of this alloy cluster; neither committee gets a pass there
(results/eos.json).

The NVE stability gate (configs/nve.yaml, from-scratch velocity-Verlet on
the active committee mean, 16-atom TiZrNb, 2 fs steps, 3 ps): total-energy
drift -0.0003 meV/atom/ps from 300 K and -0.0001 from 900 K. Whatever else
is wrong with the active committee, its forces are exact derivatives of a
smooth energy.

![NVE drift](figures/nve.png)

## Timings and scale

- Dataset: 1,488 labeled frames (112 generation groups), 0.086 s/frame
  teacher labeling on CPU, chgnet 0.4.2 chosen over MACE-MP-0 small by
  measured speed (0.180 vs 0.616 s per 54-atom call,
  data/teacher_choice.json).
- One active-learning run (7 committee trainings of 5 members plus pool
  scoring and evaluation): 20 to 22 minutes on CPU. Measured totals: main
  benchmark 3.14 h (9 runs), scarce 1.80 h (4 runs), MD exploration 27 min,
  ceiling 11 min.
- Committee member: 2,355 parameters; a committee is 11,775 parameters.

## Mid-build corrections, disclosed

- **Equal-effort epoch rule.** The first benchmark run trained every round
  for a fixed 40 epochs, which gives a 60-frame round only about 160
  optimizer steps and made the curves noisy and undertrained (a budget-100
  committee scored worse than budget-60). That run was discarded before
  anything was committed; the rule min(120, max(40, 12000/n)) epochs on n
  frames was added, applied identically to every strategy, and the entire
  benchmark rerun. The negative headline is not a training-starvation
  artifact: under the rule, a budget-60 committee improved from 266 to
  about 197 meV/A while the ranking against random stayed the same.
- **The capped variant and the scarce pool were added mid-build**, after
  the first completed runs showed pure top-k losing, to test whether the
  failure was within-batch redundancy (it is not; the cap changes nothing)
  and whether the conclusion is regime-specific (it is; see the scarce
  section). Both are reported in full rather than only the variant that
  flattered the method.

## Limitations, and what this does not show

- One alloy family, one architecture at one small capacity, one teacher,
  pool-based selection with batch acquisition. The regime conclusions
  (redundant pool: random wins; scarce pool: about 2x label saving) are
  measured facts here and qualitative guidance elsewhere.
- Acquisition variants beyond pure top-k and a per-group cap (density
  weighting, expected aggregate-error reduction, batch-diverse selection)
  were not benchmarked; the measured failure motivates them but this repo
  does not test them.
- All accuracies are against surrogate teacher labels. Nothing here is DFT
  accuracy, and the EOS section shows the physics consequences directly.
