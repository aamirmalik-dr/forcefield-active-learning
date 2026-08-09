"""Carve the committed sample from the full labeled dataset.

Keeps whole generation groups (never random frames) so the sample supports a
leakage-safe pool/test split for the offline demo: a subset of pool groups,
test groups, and transfer groups, capped to keep the file small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from falearn.data import load_frames, save_frames

REPO = Path(__file__).resolve().parents[1]
MAX_FRAMES_PER_GROUP = 10
N_GROUPS = {"pool_comps": 24, "transfer_comps": 6}


def main() -> int:
    frames = load_frames(REPO / "data" / "full" / "dataset.extxyz")
    rng = np.random.default_rng(0)
    groups: dict[str, list[int]] = {}
    for k, fr in enumerate(frames):
        groups.setdefault(fr.group, []).append(k)
    transfer = {"Ti2ZrNb", "TiZrNb2"}
    pool_groups = sorted(g for g in groups if frames[groups[g][0]].composition not in transfer)
    tr_groups = sorted(g for g in groups if frames[groups[g][0]].composition in transfer)
    rng.shuffle(pool_groups)
    rng.shuffle(tr_groups)
    picked: list[int] = []
    for g in sorted(pool_groups[: N_GROUPS["pool_comps"]]) + sorted(
        tr_groups[: N_GROUPS["transfer_comps"]]
    ):
        idx = np.array(groups[g])
        rng.shuffle(idx)
        picked.extend(sorted(idx[:MAX_FRAMES_PER_GROUP].tolist()))
    sample = [frames[k] for k in sorted(picked)]
    out = REPO / "data" / "sample_frames.extxyz"
    save_frames(sample, out)
    n_groups = len({fr.group for fr in sample})
    print(f"wrote {len(sample)} frames from {n_groups} groups -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
