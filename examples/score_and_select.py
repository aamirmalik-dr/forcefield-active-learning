"""Score the committed sample with the committed committee and pick queries.

Shows the two core library moves outside the benchmark harness: computing
committee force disagreement for candidate structures, and selecting a
query batch, all offline on committed files.

Run from the repo root:
    python examples/score_and_select.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from falearn import (
    AcsfParams,
    committee_predict,
    frame_graph,
    load_frames,
    load_members,
    select_top_disagreement,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    members, extra = load_members(REPO / "models" / "committee_active.pt")
    frames = load_frames(REPO / "data" / "sample_frames.extxyz")
    acsf = AcsfParams()
    graphs = [frame_graph(fr, acsf) for fr in frames]
    pred = committee_predict(members, graphs)
    scores = pred["disagreement_mean"] * 1000  # meV/A
    print(
        f"committee of {len(members)} members (trained at budget "
        f"{extra.get('budget', '?')}), {len(frames)} candidate frames"
    )
    print(
        f"disagreement: median {np.median(scores):.1f}, "
        f"90th percentile {np.percentile(scores, 90):.1f}, "
        f"max {scores.max():.1f} meV/A"
    )
    pick = select_top_disagreement(np.arange(len(frames)), scores, 5)
    print("\nthe 5 frames this committee would send to the teacher next:")
    for k in pick:
        fr = frames[int(k)]
        print(
            f"  frame {int(k):4d}  {fr.composition:<8} {fr.group:<24} "
            f"{fr.n_atoms:>3} atoms  disagreement {scores[int(k)]:.1f} meV/A"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
