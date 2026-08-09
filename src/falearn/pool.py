"""The candidate pool with hidden labels and an explicit purchase step.

Budget-accounting integrity is enforced structurally: the pool exposes only
structures (positions, cells, species, group metadata) to the selection code.
Labels live in a private store and become readable only after purchase(),
which is also the only place the label counter increments. Frames added later
(e.g. ensemble-MD exploration candidates) have no stored label; purchasing
one calls the oracle, which is a real teacher call.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from falearn.data import Frame, group_kind


class LabeledPool:
    """Pool of candidate structures whose labels are hidden until purchased.

    Args:
        frames: Labeled frames forming the initial pool. Labels are moved to
            the private store; the public frames carry energy=None.
        oracle: Optional callable frame -> (energy, forces) used to label
            frames that have no stored label (exploration candidates). Each
            oracle call is counted separately in oracle_calls.
    """

    def __init__(
        self,
        frames: list[Frame],
        oracle: Callable[[Frame], tuple[float, np.ndarray]] | None = None,
    ) -> None:
        self._store: dict[int, tuple[float, np.ndarray]] = {}
        self.frames: list[Frame] = []
        self._oracle = oracle
        self._purchased: list[int] = []
        self.oracle_calls = 0
        for fr in frames:
            self.add_frame(fr)

    def add_frame(self, fr: Frame) -> int:
        """Add one candidate; strips and stores any label it carries."""
        k = len(self.frames)
        if fr.energy is not None:
            self._store[k] = (float(fr.energy), np.array(fr.forces))
        self.frames.append(
            Frame(
                species=fr.species.copy(),
                positions=fr.positions.copy(),
                cell=fr.cell.copy(),
                energy=None,
                forces=None,
                group=fr.group,
                composition=fr.composition,
            )
        )
        return k

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def purchased(self) -> list[int]:
        """Indices purchased so far, in purchase order."""
        return list(self._purchased)

    @property
    def n_purchased(self) -> int:
        return len(self._purchased)

    def available(self) -> np.ndarray:
        """Indices not yet purchased."""
        bought = set(self._purchased)
        return np.array([k for k in range(len(self.frames)) if k not in bought])

    def purchase(self, indices) -> None:
        """Buy labels for the given frame indices (one budget unit each)."""
        for k in indices:
            k = int(k)
            if k in set(self._purchased):
                raise ValueError(f"frame {k} was already purchased")
            if k not in self._store:
                if self._oracle is None:
                    raise KeyError(f"frame {k} has no stored label and no oracle is set")
                energy, forces = self._oracle(self.frames[k])
                self._store[k] = (float(energy), np.array(forces))
                self.oracle_calls += 1
            self._purchased.append(k)

    def label(self, k: int) -> tuple[float, np.ndarray]:
        """The label of a PURCHASED frame; raises for unpurchased frames."""
        if k not in set(self._purchased):
            raise KeyError(f"frame {k} has not been purchased; its label is hidden")
        return self._store[k]

    def purchased_labels(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Energies, forces, and element counts of all purchased frames."""
        energies = np.array([self._store[k][0] for k in self._purchased])
        forces = [self._store[k][1] for k in self._purchased]
        counts = np.stack([self.frames[k].counts() for k in self._purchased])
        return energies, forces, counts

    def summary_row(self, k: int) -> dict:
        """Structure-only metadata of one frame (safe for selection code)."""
        fr = self.frames[k]
        return {
            "index": k,
            "group": fr.group,
            "composition": fr.composition,
            "kind": group_kind(fr.group, fr.composition),
            "n_atoms": fr.n_atoms,
        }
