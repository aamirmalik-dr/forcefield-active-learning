"""BCC TiZrNb structure generation: supercells, rattle and strain batches.

Compositions are realized as random site occupancy on an ideal BCC lattice at
a Vegard-rule lattice constant. Every batch of frames carries a generation
group id so pool/test splits can respect provenance. Candidate generation is
teacher-free on purpose: no teacher-driven dynamics shapes the pool, so the
labeling budget accounting stays clean.
"""

from __future__ import annotations

import numpy as np

from falearn.data import Frame

# Approximate BCC lattice constants in Angstrom used only to seed generation
# (PBE-DFT-range values; benchmarks measure actual minima later).
BCC_A = {"Ti": 3.25, "Zr": 3.57, "Nb": 3.31}

COMPOSITIONS: dict[str, tuple[float, float, float]] = {
    "Ti": (1.0, 0.0, 0.0),
    "Zr": (0.0, 1.0, 0.0),
    "Nb": (0.0, 0.0, 1.0),
    "TiZr": (0.5, 0.5, 0.0),
    "TiNb": (0.5, 0.0, 0.5),
    "ZrNb": (0.0, 0.5, 0.5),
    "TiZrNb": (1 / 3, 1 / 3, 1 / 3),
    "Ti2ZrNb": (0.5, 0.25, 0.25),
    "TiZrNb2": (0.25, 0.25, 0.5),
}

POOL_COMPOSITIONS = ("Ti", "Zr", "Nb", "TiZr", "TiNb", "ZrNb", "TiZrNb")
TRANSFER_COMPOSITIONS = ("Ti2ZrNb", "TiZrNb2")


def vegard_a(composition: str) -> float:
    """Composition-weighted BCC lattice constant (Vegard rule)."""
    frac = COMPOSITIONS[composition]
    return float(sum(f * BCC_A[el] for f, el in zip(frac, ("Ti", "Zr", "Nb"))))


def bcc_supercell(composition: str, reps: int, seed: int) -> Frame:
    """Ideal BCC supercell with random site occupancy.

    Args:
        composition: Key into COMPOSITIONS.
        reps: Cubic repetitions; atoms = 2 * reps**3.
        seed: RNG seed for the site-occupancy shuffle.

    Returns:
        Unlabeled Frame at the ideal Vegard-rule geometry.
    """
    rng = np.random.default_rng(seed)
    a = vegard_a(composition)
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    frac_pos = []
    for ix in range(reps):
        for iy in range(reps):
            for iz in range(reps):
                frac_pos.extend((basis + np.array([ix, iy, iz])) / reps)
    frac_pos = np.array(frac_pos)
    n_atoms = len(frac_pos)
    frac = COMPOSITIONS[composition]
    counts = np.floor(np.array(frac) * n_atoms).astype(int)
    while counts.sum() < n_atoms:
        counts[np.argmax(np.array(frac) - counts / n_atoms)] += 1
    species = np.repeat(np.arange(3), counts)
    rng.shuffle(species)
    cell = np.eye(3) * a * reps
    return Frame(
        species=species,
        positions=frac_pos @ cell,
        cell=cell,
        composition=composition,
    )


def rattle_batch(
    composition: str, reps: int, amplitude: float, n_frames: int, seed: int, batch: int
) -> list[Frame]:
    """Frames with Gaussian displacements from independent random occupancies."""
    frames = []
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        fr = bcc_supercell(composition, reps, seed=int(rng.integers(2**31)))
        fr.positions = fr.positions + rng.normal(0.0, amplitude, fr.positions.shape)
        fr.group = f"{composition}_rattle{int(round(amplitude * 100)):02d}_b{batch}"
        frames.append(fr)
    return frames


def strain_batch(
    composition: str,
    reps: int,
    n_frames: int,
    seed: int,
    batch: int,
    max_volumetric: float = 0.06,
    max_shear: float = 0.05,
    jitter: float = 0.03,
) -> list[Frame]:
    """Volumetric and shear strained frames with a small symmetry-breaking rattle."""
    frames = []
    rng = np.random.default_rng(seed)
    for k in range(n_frames):
        fr = bcc_supercell(composition, reps, seed=int(rng.integers(2**31)))
        if k % 2 == 0:
            scale = 1.0 + rng.uniform(-max_volumetric, max_volumetric)
            strain = np.eye(3) * scale
            tag = "vol"
        else:
            strain = np.eye(3)
            pairs = [(0, 1), (0, 2), (1, 2)]
            i, j = pairs[rng.integers(0, 3)]
            strain[i, j] = rng.uniform(-max_shear, max_shear)
            tag = "shear"
        frac = fr.positions @ np.linalg.inv(fr.cell)
        fr.cell = fr.cell @ strain.T
        fr.positions = frac @ fr.cell + rng.normal(0.0, jitter, fr.positions.shape)
        fr.group = f"{composition}_strain_{tag}_b{batch}"
        frames.append(fr)
    return frames


def build_dataset(
    compositions: tuple[str, ...],
    rattle_amplitudes: tuple[float, ...] = (0.05, 0.10, 0.18, 0.26),
    batches_per_kind: int = 2,
    frames_per_batch: int = 14,
    reps_cycle: tuple[int, ...] = (2, 2, 2, 3),
    seed: int = 20260809,
) -> list[Frame]:
    """The full unlabeled candidate/test population for one composition set.

    Each (composition, rattle amplitude or strain, batch index) triple becomes
    one generation group. Supercell sizes cycle through reps_cycle across
    batches so the population mixes 16-atom and 54-atom cells.

    Returns:
        Unlabeled frames; label with falearn.teacher.label_frames.
    """
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    for comp in compositions:
        cycle = 0
        for amp in rattle_amplitudes:
            for b in range(batches_per_kind):
                reps = reps_cycle[cycle % len(reps_cycle)]
                cycle += 1
                frames.extend(
                    rattle_batch(
                        comp,
                        reps,
                        amp,
                        frames_per_batch,
                        seed=int(rng.integers(2**31)),
                        batch=b,
                    )
                )
        for b in range(batches_per_kind):
            reps = reps_cycle[cycle % len(reps_cycle)]
            cycle += 1
            frames.extend(
                strain_batch(comp, reps, frames_per_batch, seed=int(rng.integers(2**31)), batch=b)
            )
    return frames
