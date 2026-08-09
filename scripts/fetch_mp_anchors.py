"""Fetch DFT validation anchors from the Materials Project API.

Materials Project data is CC BY 4.0 and requires a free API key in the
MP_API_KEY environment variable. When a key is configured this script writes
data/anchors_mp.json with BCC lattice constants and bulk moduli for Ti, Zr,
Nb. Without a key it exits cleanly and the EOS benchmark falls back to
data/anchors_literature.json, whose values carry citations.

These anchors are the only real DFT numbers in this project. Training labels
come from a surrogate universal potential, never from DFT.

Usage:
    MP_API_KEY=... python scripts/fetch_mp_anchors.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        print("MP_API_KEY not set; using committed literature anchors instead.")
        print("Get a free key at https://next-gen.materialsproject.org/api and re-run.")
        return 0
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("mp-api not installed; pip install 'falearn[mp]' first.", file=sys.stderr)
        return 1

    anchors: dict = {
        "source": "Materials Project API (CC BY 4.0)",
        "note": "DFT (GGA/GGA+U workflow) reference values for BCC phases",
        "entries": {},
    }
    with MPRester(key) as mpr:
        for formula in ("Ti", "Zr", "Nb"):
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "structure", "symmetry", "bulk_modulus"],
            )
            for doc in docs:
                if doc.symmetry is None or doc.symmetry.symbol != "Im-3m":
                    continue
                st = doc.structure
                # the API returns the primitive cell; recover the conventional
                # BCC lattice constant from the volume per atom (2 atoms per
                # conventional cell)
                v_atom = float(st.volume) / len(st)
                entry = {
                    "material_id": str(doc.material_id),
                    "a0_bcc_a": round((2.0 * v_atom) ** (1.0 / 3.0), 4),
                    "volume_per_atom_a3": round(v_atom, 4),
                    "spacegroup": "Im-3m",
                }
                bulk = doc.bulk_modulus
                vrh = None
                if isinstance(bulk, dict):
                    vrh = bulk.get("vrh")
                elif bulk is not None:
                    vrh = getattr(bulk, "vrh", None)
                if vrh is not None:
                    entry["b0_gpa"] = float(vrh)
                anchors["entries"][formula] = entry
                break
    out = REPO / "data" / "anchors_mp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anchors, indent=2))
    print(f"wrote {out} with {len(anchors['entries'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
