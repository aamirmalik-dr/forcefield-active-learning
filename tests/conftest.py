"""Shared test fixtures: tiny frames and analytically labeled datasets."""

from __future__ import annotations

import numpy as np
import pytest

from falearn.data import Frame
from falearn.descriptors import AcsfParams, frame_graph
from falearn.model import MemberPotential
from falearn.structures import bcc_supercell, rattle_batch


@pytest.fixture(scope="session")
def small_acsf() -> AcsfParams:
    return AcsfParams()


@pytest.fixture()
def tizrnb_frame() -> Frame:
    fr = bcc_supercell("TiZrNb", 2, seed=3)
    rng = np.random.default_rng(5)
    fr.positions = fr.positions + rng.normal(0, 0.08, fr.positions.shape)
    fr.group = "TiZrNb_rattle08_b0"
    return fr


def oracle_potential(acsf: AcsfParams) -> MemberPotential:
    """A fixed untrained member acting as a smooth synthetic ground truth."""
    m = MemberPotential(acsf, hidden=(8,), seed=99)
    m.set_reference(np.array([-5.0, -6.0, -7.0]))
    m.eval()
    return m


def label_with_member(frames: list[Frame], member: MemberPotential, acsf: AcsfParams) -> None:
    """Label frames in place using a member potential as synthetic teacher."""
    for fr in frames:
        graph = frame_graph(fr, acsf)
        energies, forces = member.energy_forces(graph)
        fr.energy = float(energies[0].detach())
        fr.forces = forces.detach().numpy()


@pytest.fixture()
def labeled_batches(small_acsf) -> list[Frame]:
    """Six tiny generation groups, labeled by the synthetic oracle."""
    oracle = oracle_potential(small_acsf)
    frames: list[Frame] = []
    comp_id = {"TiZrNb": 0, "TiNb": 1, "ZrNb": 2}
    pairs = [("TiZrNb", 0), ("TiZrNb", 1), ("TiNb", 0), ("TiNb", 1), ("ZrNb", 0), ("ZrNb", 1)]
    for comp, batch in pairs:
        frames.extend(rattle_batch(comp, 2, 0.08, 6, seed=batch * 7 + comp_id[comp], batch=batch))
    label_with_member(frames, oracle, small_acsf)
    return frames
