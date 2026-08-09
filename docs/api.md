# falearn API

The package is a small library first and a CLI second. Everything the CLI
does goes through these public functions.

## Data model

```python
from falearn import Frame, load_frames, save_frames

frames = load_frames("data/sample_frames.extxyz")
fr = frames[0]
fr.species      # (N,) int array, 0=Ti 1=Zr 2=Nb
fr.positions    # (N, 3) Cartesian Angstrom
fr.cell         # (3, 3) row-vector lattice
fr.energy       # eV, surrogate teacher label (None if unlabeled)
fr.forces       # (N, 3) eV/A
fr.group        # generation-group id, e.g. "TiZrNb_rattle14_b0"
```

Splits operate on whole generation groups and whole compositions:

```python
from falearn import pool_test_split

split = pool_test_split(frames, holdout_compositions=("Ti2ZrNb", "TiZrNb2"),
                        test_fraction=0.2, seed=20260809)
split.pool, split.test, split.transfer   # disjoint index lists
```

## The pool with hidden labels

`LabeledPool` enforces the budget accounting structurally: labels are moved
into a private store at construction and become readable only after
`purchase()`.

```python
from falearn import LabeledPool

pool = LabeledPool([frames[k] for k in split.pool])
pool.frames[0].energy        # None: selection code cannot see labels
pool.purchase([3, 17, 42])   # three budget units
energies, forces, counts = pool.purchased_labels()
pool.label(3)                # fine
pool.label(4)                # KeyError: not purchased
```

An optional `oracle` callable labels frames that have no stored label (MD
exploration candidates); each oracle call is a real teacher call and is
counted in `pool.oracle_calls`.

## Committee and disagreement

```python
from falearn import AcsfParams, MemberPotential, train_committee, committee_predict

acsf = AcsfParams()                      # 30 features, rc 5.5 / 4.5 A
members, hist = train_committee(
    n_members=5, graphs=graphs, energies=energies, forces=forces,
    counts=counts, settings=train_settings, acsf=acsf, hidden=(16, 16),
    base_seed=0,
)
pred = committee_predict(members, graphs)
pred["disagreement_mean"]   # per-frame committee force disagreement (eV/A)
pred["e_mean"]              # committee-mean energies (eV)
```

The disagreement of atom i is `sqrt(mean_m |F_m,i - Fbar_i|^2)`, the RMS
deviation of member force vectors from the committee mean; the per-frame
score is its mean over atoms.

## The loop

```python
from falearn import LoopSettings, TrainSettings, EvalSet, run_loop

settings = LoopSettings(strategy="committee", n_members=5, n_init=60,
                        round_size=40, n_rounds=6, seed=0,
                        train=TrainSettings(epochs=40, batch_size=16))
history, members = run_loop(pool, [test_set], settings, acsf=acsf)
history["rounds"][k]["budget"]              # labels spent by round k
history["rounds"][k]["metrics"]["test"]     # committee-mean errors
history["rounds"][k]["selected"]            # what was bought next, with scores
history["calibration"]                      # per-frame disagreement vs error
```

`strategy="random"` runs the baseline: identical initial draw, identical
training, uniform random purchases.

## Physics checks

```python
from falearn.eos import eos_scan_committee, fit_bm
from falearn.md import nve_run, langevin_snapshots
```

`nve_run` integrates velocity-Verlet NVE with committee-mean forces and
reports the drift in meV/atom/ps; `langevin_snapshots` is the BAOAB sampler
behind MD exploration.

## Benchmarks

```python
from falearn.bench import load_config, run_al_benchmark

results = run_al_benchmark(load_config("configs/fast.yaml"))
```

See `configs/*.yaml` for every knob. The CLI wraps these:

```
falearn demo                      # offline micro loop on the committed sample
falearn run --config configs/main.yaml
falearn eval --committee models/committee_active.pt --frames data/sample_frames.extxyz
falearn generate                  # rebuild and relabel the full dataset (needs chgnet)
```
