"""Member potential: force correctness, reproducibility, serialization."""

from __future__ import annotations

import numpy as np
import torch

from falearn.descriptors import frame_graph
from falearn.model import MemberPotential, load_members, save_members


def test_forces_match_finite_differences(tizrnb_frame, small_acsf):
    member = MemberPotential(small_acsf, hidden=(8,), seed=0)
    member.set_reference(np.array([-5.0, -6.0, -7.0]))
    member.eval()
    graph = frame_graph(tizrnb_frame, small_acsf)
    _, forces = member.energy_forces(graph)
    forces = forces.numpy()
    h = 1e-6
    rng = np.random.default_rng(0)
    for _ in range(4):
        a = int(rng.integers(tizrnb_frame.n_atoms))
        c = int(rng.integers(3))
        for sign, store in ((1, "plus"), (-1, "minus")):
            fr = tizrnb_frame
            pos = fr.positions.copy()
            pos[a, c] += sign * h
            moved = frame_graph(
                type(fr)(species=fr.species, positions=pos, cell=fr.cell), small_acsf
            )
            with torch.no_grad():
                e = float(member(moved.positions, moved)[0])
            if store == "plus":
                e_plus = e
            else:
                e_minus = e
        fd = -(e_plus - e_minus) / (2 * h)
        assert abs(fd - forces[a, c]) < 1e-6


def test_member_seed_reproducibility(small_acsf):
    a = MemberPotential(small_acsf, seed=7)
    b = MemberPotential(small_acsf, seed=7)
    c = MemberPotential(small_acsf, seed=8)
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    pc = torch.cat([p.flatten() for p in c.parameters()])
    assert torch.equal(pa, pb)
    assert not torch.equal(pa, pc)


def test_save_load_round_trip(tmp_path, tizrnb_frame, small_acsf):
    members = [MemberPotential(small_acsf, hidden=(8,), seed=k) for k in range(3)]
    for m in members:
        m.set_reference(np.array([-5.0, -6.0, -7.0]))
    path = tmp_path / "committee.pt"
    save_members(members, path, extra={"budget": 42})
    loaded, extra = load_members(path)
    assert extra["budget"] == 42
    assert len(loaded) == 3
    graph = frame_graph(tizrnb_frame, small_acsf)
    for orig, new in zip(members, loaded):
        e0, f0 = orig.energy_forces(graph)
        e1, f1 = new.energy_forces(graph)
        assert torch.allclose(e0, e1, atol=0.0)
        assert torch.allclose(f0, f1, atol=0.0)


def test_parameter_count_is_compact(small_acsf):
    member = MemberPotential(small_acsf, hidden=(16, 16))
    assert member.n_parameters < 5000
