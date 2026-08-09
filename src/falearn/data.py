"""Frame container, extxyz persistence, and leakage-safe pool/test splits.

A Frame is one periodic structure with optional teacher labels. Every frame
carries a generation-group id (one rattle or strain batch, or one MD
trajectory) and a composition tag. The candidate pool and the test sets are
split by whole generation groups and by whole compositions, never by random
frames, so sibling frames from one batch can never sit on both sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SYMBOL_TO_INDEX = {"Ti": 0, "Zr": 1, "Nb": 2}
INDEX_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_INDEX.items()}


@dataclass
class Frame:
    """One periodic structure with optional labels.

    Attributes:
        species: (N,) int array, 0=Ti 1=Zr 2=Nb.
        positions: (N, 3) Cartesian Angstrom.
        cell: (3, 3) row-vector lattice.
        energy: Total energy in eV (surrogate teacher label), or None.
        forces: (N, 3) eV/Angstrom (surrogate teacher label), or None.
        group: Generation-group id, e.g. "TiZrNb_rattle10_b0".
        composition: Composition tag, e.g. "TiZrNb".
    """

    species: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    group: str = ""
    composition: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    def counts(self) -> np.ndarray:
        """Per-element atom counts, shape (3,)."""
        return np.bincount(self.species, minlength=3)


def save_frames(frames: list[Frame], path: str | Path) -> None:
    """Write frames to an extxyz file (via ASE, structure handling only)."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    images = []
    for fr in frames:
        atoms = Atoms(
            symbols=[INDEX_TO_SYMBOL[int(s)] for s in fr.species],
            positions=fr.positions,
            cell=fr.cell,
            pbc=True,
        )
        atoms.info["group"] = fr.group
        atoms.info["composition"] = fr.composition
        if fr.energy is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=fr.energy, forces=fr.forces)
        images.append(atoms)
    write(str(path), images, format="extxyz")


def load_frames(path: str | Path) -> list[Frame]:
    """Read frames from an extxyz file written by save_frames."""
    from ase.io import read

    frames = []
    for atoms in read(str(path), index=":", format="extxyz"):
        energy = None
        forces = None
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.array(atoms.get_forces())
            except Exception:
                energy, forces = None, None
        frames.append(
            Frame(
                species=np.array([SYMBOL_TO_INDEX[s] for s in atoms.get_chemical_symbols()]),
                positions=np.array(atoms.get_positions()),
                cell=np.array(atoms.get_cell()),
                energy=energy,
                forces=forces,
                group=str(atoms.info.get("group", "")),
                composition=str(atoms.info.get("composition", "")),
            )
        )
    return frames


def group_kind(group: str, composition: str) -> str:
    """The batch type of a generation-group id, e.g. 'rattle18' or 'strain_vol'.

    Group ids follow '<composition>_<kind>_b<batch>'; the kind is what remains
    after stripping the composition prefix and the batch suffix.
    """
    kind = group
    if composition and kind.startswith(composition + "_"):
        kind = kind[len(composition) + 1 :]
    return re.sub(r"_b\d+$", "", kind)


@dataclass
class PoolSplit:
    """Index split of a labeled frame list into pool and test sets.

    Attributes:
        pool: Candidate-pool indices (labels hidden until purchased).
        test: Held-out-group test indices (same compositions as the pool).
        transfer: Held-out-composition test indices.
    """

    pool: list[int]
    test: list[int]
    transfer: list[int] = field(default_factory=list)


def pool_test_split(
    frames: list[Frame],
    holdout_compositions: tuple[str, ...] = ("Ti2ZrNb", "TiZrNb2"),
    test_fraction: float = 0.2,
    seed: int = 0,
) -> PoolSplit:
    """Leakage-safe split: whole generation groups, and whole compositions.

    Frames whose composition is in holdout_compositions go entirely to the
    transfer set. The remaining generation groups are stratified by batch
    type (rattle amplitude, strain kind) so the test set's difficulty mix
    matches the pool; a test_fraction of the groups of each type is held out.

    Args:
        frames: Labeled frames.
        holdout_compositions: Composition tags reserved for transfer testing.
        test_fraction: Fraction of in-distribution groups per batch type held
            out for testing.
        seed: RNG seed for the per-type group shuffle.

    Returns:
        PoolSplit with disjoint pool/test group sets and a transfer set.
    """
    rng = np.random.default_rng(seed)
    transfer = [k for k, fr in enumerate(frames) if fr.composition in holdout_compositions]
    in_dist = [k for k, fr in enumerate(frames) if fr.composition not in holdout_compositions]
    kind_of = {}
    for k in in_dist:
        kind_of[frames[k].group] = group_kind(frames[k].group, frames[k].composition)
    by_kind: dict[str, list[str]] = {}
    for g in sorted(kind_of):
        by_kind.setdefault(kind_of[g], []).append(g)
    test_groups: set[str] = set()
    for kind in sorted(by_kind):
        groups = by_kind[kind]
        rng.shuffle(groups)
        n_test = max(1, int(round(test_fraction * len(groups))))
        test_groups.update(groups[:n_test])
    split = PoolSplit(pool=[], test=[], transfer=transfer)
    for k in in_dist:
        if frames[k].group in test_groups:
            split.test.append(k)
        else:
            split.pool.append(k)
    return split


def sibling_test_indices(
    frames: list[Frame],
    pool_indices: list[int],
    purchased: list[int],
    n_per_group: int = 2,
    seed: int = 0,
) -> list[int]:
    """Unpurchased pool frames from purchased groups: the leaky test set.

    A random-frame protocol would happily test on frames like these, siblings
    of training frames from the same rattle or MD batch. Used once, to show
    what that protocol would have claimed; never for reported model quality.
    """
    rng = np.random.default_rng(seed)
    purchased_set = set(purchased)
    purchased_groups = {frames[k].group for k in purchased}
    by_group: dict[str, list[int]] = {}
    for k in pool_indices:
        if k not in purchased_set and frames[k].group in purchased_groups:
            by_group.setdefault(frames[k].group, []).append(k)
    picked: list[int] = []
    for g in sorted(by_group):
        idx = np.array(by_group[g])
        rng.shuffle(idx)
        picked.extend(idx[:n_per_group].tolist())
    return sorted(picked)
